# JMM MCP

Custom MCP server for the Jeremy Management Model. Runs as a Supabase Edge Function.

## Setup on a new machine

1. Install Supabase CLI: brew install supabase/tap/supabase
2. Clone this repo: git clone <repo-url>
3. Link to project: supabase link --project-ref <project-ref>
4. Deploy: supabase functions deploy jmm-mcp

## Structure

- supabase/functions/jmm-mcp/index.ts — the MCP server
- supabase/config.toml — Supabase project config
