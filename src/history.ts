import type { SessionClient, TranscriptTurn } from "./session-client.js";

// session-manager caps one window at 1000 rows. Page backwards until the
// entire private provider history is available before resuming a thinking
// session; a repeated or empty page is an error, not partial history.
export async function loadCompleteTranscript(client: SessionClient, sessionId: string): Promise<TranscriptTurn[] | null> {
  const pages: TranscriptTurn[][] = [];
  let beforeSeq = 0;
  for (let pageIndex = 0; pageIndex < 10_000; pageIndex++) {
    const page = await client.getTranscript(sessionId, { limit: 1000, beforeSeq });
    if (!page) return null;
    const firstSeq = page.messages[0]?.seq;
    if (page.hasMore && (firstSeq === undefined || (beforeSeq > 0 && firstSeq >= beforeSeq))) return null;
    pages.unshift(page.messages);
    if (!page.hasMore) return pages.flat();
    beforeSeq = firstSeq!;
  }
  return null;
}
