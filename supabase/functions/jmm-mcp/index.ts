// JMM MCP — Custom logic for the Jeremy Management Model
// Generic CRUD is handled by the native Supabase MCP.
// This function enforces JMM-specific rules: memory approval, triage queries,
// and the memory semantic layer — embeddings (Supabase.ai gte-small, 384 dims),
// hybrid vector + keyword recall, near-duplicate detection, and embedding backfill.
// It is also the embedding service the Node JMM servers (jmm-mcp-server) call
// via the embed_text tool.

import "@supabase/functions-js/edge-runtime.d.ts"
import { createClient } from "jsr:@supabase/supabase-js@2"

const supabaseUrl = Deno.env.get("SUPABASE_URL")!
const supabaseServiceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
const supabase = createClient(supabaseUrl, supabaseServiceKey)

interface ToolRequest {
  tool: string
  args: Record<string, unknown>
}

Deno.serve(async (req) => {
  try {
    const body: ToolRequest = await req.json()
    const { tool, args } = body

    let result

    switch (tool) {
      case "add_jeremy_memory":
        result = await addJeremyMemory(args)
        break
      case "add_claude_memory":
        result = await addClaudeMemory(args)
        break
      case "recall_memory":
        result = await recallMemory(args)
        break
      case "embed_text":
        result = await embedText(args)
        break
      case "backfill_memory_embeddings":
        result = await backfillMemoryEmbeddings(args)
        break
      case "get_priorities":
        result = await getPriorities(args)
        break
      case "get_stalled":
        result = await getStalled(args)
        break
      default:
        return jsonError(`Unknown tool: ${tool}`, 400)
    }

    return jsonResponse(result)
  } catch (err) {
    return jsonError(err instanceof Error ? err.message : "Unknown error", 500)
  }
})

// ---------- Embeddings (Supabase.ai, built into the edge runtime) ----------

const EMBEDDING_MODEL = "gte-small"
const EMBEDDING_DIMENSIONS = 384
const MAX_EMBED_TEXTS = 32
const MAX_EMBED_CHARS = 8000
// Cosine similarity at/above which an existing memory is flagged as a likely duplicate.
const DUPLICATE_SIMILARITY = 0.86

interface AiSession {
  run(input: string, opts: { mean_pool: boolean; normalize: boolean }): Promise<number[]>
}

let gteSession: AiSession | null = null

function getEmbeddingSession(): AiSession {
  if (!gteSession) {
    const ai = (globalThis as unknown as {
      Supabase?: { ai?: { Session: new (model: string) => AiSession } }
    }).Supabase?.ai
    if (!ai) throw new Error("Supabase.ai is not available in this runtime")
    gteSession = new ai.Session(EMBEDDING_MODEL)
  }
  return gteSession
}

async function embedTexts(texts: string[]): Promise<number[][]> {
  const session = getEmbeddingSession()
  const out: number[][] = []
  for (const text of texts) {
    const vec = await session.run(text.slice(0, MAX_EMBED_CHARS), {
      mean_pool: true,
      normalize: true,
    })
    out.push(Array.from(vec))
  }
  return out
}

async function bestEffortEmbed(text: string): Promise<number[] | null> {
  try {
    const [vec] = await embedTexts([text])
    return vec
  } catch (_err) {
    return null
  }
}

async function embedText(args: Record<string, unknown>) {
  const { texts } = args as { texts?: unknown }
  if (!Array.isArray(texts) || texts.length === 0 || !texts.every((t) => typeof t === "string")) {
    throw new Error("texts is required: a non-empty array of strings")
  }
  if (texts.length > MAX_EMBED_TEXTS) {
    throw new Error(`texts is limited to ${MAX_EMBED_TEXTS} items per call`)
  }
  const embeddings = await embedTexts(texts as string[])
  return { model: EMBEDDING_MODEL, dimensions: EMBEDDING_DIMENSIONS, embeddings }
}

// ---------- Memory helpers ----------

const MEMORY_COLUMNS =
  "id, layer, content, tags, source_context, importance, status, reference_count, created_at, last_referenced"
const MEMORY_COLUMNS_LEGACY = "id, layer, content, tags, source_context, created_at, last_referenced"

interface SimilarMemory {
  id: string
  layer: string
  content: string
  tags: string[]
  similarity: number
  created_at: string
}

async function findLikelyDuplicates(embedding: number[], layer: string): Promise<SimilarMemory[]> {
  const { data, error } = await supabase.rpc("memory_find_similar", {
    p_embedding: embedding,
    p_layer: layer,
    p_limit: 3,
    p_exclude: null,
  })
  if (error) return [] // migration not applied yet — dup check is best-effort
  return ((data ?? []) as SimilarMemory[]).filter((r) => r.similarity >= DUPLICATE_SIMILARITY)
}

// Insert with the semantic columns, falling back to the legacy shape when the
// memory_semantic_search migration has not been applied yet.
async function insertMemoryRow(
  base: Record<string, unknown>,
  semantic: Record<string, unknown>,
) {
  const full = await supabase
    .from("memory")
    .insert({ ...base, ...semantic })
    .select(MEMORY_COLUMNS)
    .single()
  if (!full.error) return { memory: full.data, semantic_applied: true }

  const legacy = await supabase.from("memory").insert(base).select(MEMORY_COLUMNS_LEGACY).single()
  if (legacy.error) throw legacy.error
  return { memory: legacy.data, semantic_applied: false }
}

function embeddingNote(embedding: number[] | null, applied: boolean) {
  if (!embedding) {
    return { embedded: false, note: "embedding unavailable (Supabase.ai error); run backfill_memory_embeddings later" }
  }
  if (!applied) {
    return { embedded: false, note: "semantic columns missing — apply the memory_semantic_search migration, then run backfill_memory_embeddings" }
  }
  return { embedded: true, model: EMBEDDING_MODEL }
}

// ---------- Tool implementations ----------

async function addJeremyMemory(args: Record<string, unknown>) {
  const { content, tags, source_context, approved, importance } = args as {
    content: string
    tags?: string[]
    source_context?: string
    approved?: boolean
    importance?: number
  }

  if (!content) {
    throw new Error("content is required")
  }

  const embedding = await bestEffortEmbed(content)
  const duplicates = embedding ? await findLikelyDuplicates(embedding, "jeremy") : []

  if (!approved) {
    return {
      status: "approval_required",
      message: "Jeremy memory writes require explicit approval. Summarize and ask before retrying with approved: true.",
      proposed: { content, tags, source_context, importance: importance ?? 3 },
      ...(duplicates.length
        ? {
            possible_duplicates: duplicates,
            note: "Existing memories look very similar. Consider updating or archiving one instead of writing a duplicate.",
          }
        : {}),
    }
  }

  const semantic: Record<string, unknown> = {}
  if (importance !== undefined) semantic.importance = importance
  if (embedding) {
    semantic.embedding = embedding
    semantic.embedding_model = EMBEDDING_MODEL
  }

  const { memory, semantic_applied } = await insertMemoryRow(
    { layer: "jeremy", content, tags: tags ?? [], source_context: source_context ?? null },
    semantic,
  )

  return {
    status: "written",
    memory,
    semantic: {
      ...embeddingNote(embedding, semantic_applied),
      ...(duplicates.length ? { possible_duplicates: duplicates } : {}),
    },
  }
}

async function addClaudeMemory(args: Record<string, unknown>) {
  const { content, tags, source_context, importance } = args as {
    content: string
    tags?: string[]
    source_context?: string
    importance?: number
  }

  if (!content) {
    throw new Error("content is required")
  }

  const embedding = await bestEffortEmbed(content)
  const duplicates = embedding ? await findLikelyDuplicates(embedding, "claude") : []

  const semantic: Record<string, unknown> = {}
  if (importance !== undefined) semantic.importance = importance
  if (embedding) {
    semantic.embedding = embedding
    semantic.embedding_model = EMBEDDING_MODEL
  }

  const { memory, semantic_applied } = await insertMemoryRow(
    { layer: "claude", content, tags: tags ?? [], source_context: source_context ?? null },
    semantic,
  )

  return {
    status: "written",
    memory,
    semantic: {
      ...embeddingNote(embedding, semantic_applied),
      ...(duplicates.length ? { possible_duplicates: duplicates } : {}),
    },
    note: "Claude memory written. Review the content — if it's wrong or belongs in Jeremy memory, flag it.",
  }
}

async function recallMemory(args: Record<string, unknown>) {
  const { query, layer, tag, limit, min_score, reinforce } = args as {
    query: string
    layer?: string
    tag?: string
    limit?: number
    min_score?: number
    reinforce?: boolean
  }

  if (!query || !query.trim()) {
    throw new Error("query is required: describe the topic or question in natural language")
  }

  const embedding = await bestEffortEmbed(query)

  const { data, error } = await supabase.rpc("memory_recall", {
    p_query: query,
    p_embedding: embedding,
    p_layer: layer ?? null,
    p_tag: tag ?? null,
    p_limit: Math.min(limit ?? 8, 25),
    p_min_score: min_score ?? 0,
  })
  if (error) {
    throw new Error(`memory_recall failed (is the memory_semantic_search migration applied?): ${error.message}`)
  }

  const entries = (data ?? []) as Array<{ id: string }>
  const reinforced = entries.length > 0 && reinforce !== false
  if (reinforced) {
    await supabase.rpc("memory_mark_referenced", { p_ids: entries.map((e) => e.id) })
  }

  return {
    mode: embedding ? "hybrid" : "keyword-only",
    count: entries.length,
    query,
    entries,
    reinforced,
    ...(embedding ? {} : { note: "embedding unavailable — ran keyword-only recall" }),
  }
}

async function backfillMemoryEmbeddings(args: Record<string, unknown>) {
  const { batch_size } = args as { batch_size?: number }
  const batch = Math.min(batch_size ?? 50, 200)

  const { data, error } = await supabase
    .from("memory")
    .select("id, content")
    .is("embedding", null)
    .eq("status", "active")
    .order("created_at", { ascending: true })
    .limit(batch)
  if (error) {
    throw new Error(`backfill query failed (is the memory_semantic_search migration applied?): ${error.message}`)
  }

  let embedded = 0
  const failures: Array<{ id: string; error: string }> = []
  for (const row of data ?? []) {
    try {
      const [vec] = await embedTexts([row.content ?? ""])
      const upd = await supabase
        .from("memory")
        .update({ embedding: vec, embedding_model: EMBEDDING_MODEL })
        .eq("id", row.id)
      if (upd.error) throw new Error(upd.error.message)
      embedded++
    } catch (err) {
      failures.push({ id: row.id, error: err instanceof Error ? err.message : String(err) })
    }
  }

  const { count } = await supabase
    .from("memory")
    .select("id", { count: "exact", head: true })
    .is("embedding", null)
    .eq("status", "active")

  return {
    embedded,
    failed: failures.length,
    ...(failures.length ? { failures: failures.slice(0, 5) } : {}),
    remaining: count ?? null,
    done: (count ?? 0) === 0,
  }
}

async function getPriorities(args: Record<string, unknown>) {
  const { limit } = args as { limit?: number }
  const max = limit ?? 10

  const { data, error } = await supabase
    .from("thread")
    .select(`
      id, title, type, status, next_action, last_touched, waiting_on, notes,
      project:project_id (id, name, domain)
    `)
    .eq("type", "load-bearing")
    .eq("status", "active")
    .order("last_touched", { ascending: false })
    .limit(max)

  if (error) throw error

  return {
    count: data?.length ?? 0,
    threads: data,
  }
}

async function getStalled(args: Record<string, unknown>) {
  const { domain } = args as { domain?: string }

  let query = supabase
    .from("thread")
    .select(`
      id, title, type, status, next_action, last_touched, waiting_on, notes,
      project:project_id (id, name, domain)
    `)
    .eq("status", "stalled")
    .order("last_touched", { ascending: true })

  if (domain) {
    const { data: projects, error: projectError } = await supabase
      .from("project")
      .select("id")
      .eq("domain", domain)

    if (projectError) throw projectError

    const projectIds = projects?.map((p) => p.id) ?? []
    if (projectIds.length === 0) {
      return { count: 0, threads: [], note: `No projects in domain: ${domain}` }
    }
    query = query.in("project_id", projectIds)
  }

  const { data, error } = await query
  if (error) throw error

  return {
    count: data?.length ?? 0,
    threads: data,
    domain: domain ?? "all",
  }
}

// ---------- Helpers ----------

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body, null, 2), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function jsonError(message: string, status = 500): Response {
  return jsonResponse({ error: message }, status)
}
