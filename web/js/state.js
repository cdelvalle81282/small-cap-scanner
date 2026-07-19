// tiny shared state: the signal the user last opened, so the Detail view can
// run "Analyze with AI" with the full signal context (not just what's in the URL)
export const state = { signal: null };
