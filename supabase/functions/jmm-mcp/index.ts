// JMM MCP — Custom logic for the Jeremy Management Model
// Generic CRUD is handled by the native Supabase MCP.
// This function enforces JMM-specific rules: memory approval, triage queries.

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

// ---------- Tool implementations ----------

async function addJeremyMemory(args: Record<string, unknown>) {
  const { content, tags, source_context, approved } = args as {
    content: string
    tags?: string[]
    source_context?: string
    approved?: boolean
  }

  if (!approved) {
    return {
      status: "approval_required",
      message: "Jeremy memory writes require explicit approval. Summarize and ask before retrying with approved: true.",
      proposed: { content, tags, source_context },
    }
  }

  if (!content) {
    throw new Error("content is required")
  }

  const { data, error } = await supabase
    .from("memory")
    .insert({
      layer: "jeremy",
      content,
      tags: tags ?? [],
      source_context: source_context ?? null,
    })
    .select()
    .single()

  if (error) throw error

  return { status: "written", memory: data }
}

async function addClaudeMemory(args: Record<string, unknown>) {
  const { content, tags, source_context } = args as {
    content: string
    tags?: string[]
    source_context?: string
  }

  if (!content) {
    throw new Error("content is required")
  }

  const { data, error } = await supabase
    .from("memory")
    .insert({
      layer: "claude",
      content,
      tags: tags ?? [],
      source_context: source_context ?? null,
    })
    .select()
    .single()

  if (error) throw error

  return {
    status: "written",
    memory: data,
    note: "Claude memory written. Review the content — if it's wrong or belongs in Jeremy memory, flag it.",
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
