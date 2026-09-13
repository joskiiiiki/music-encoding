<script lang="ts">
	import { audioLabel, type AudioKind } from '$lib/api';

	let { kind, class: className = '' }: { kind: AudioKind; class?: string } = $props();

	// The distinction users need is "is this the actual corpus track, or a stand-in?" --
	// a local mp3 is the track the embedding came from, a preview is a different
	// recording found by fuzzy text search.
	const detail: Record<AudioKind, string> = {
		local: 'Full track, streaming from the local MTG tarball — this is the audio the embedding came from.',
		preview:
			'30s preview from Deezer/iTunes — a different recording of the song, matched by artist and title.',
		none: 'Neither in the local slice nor found on Deezer/iTunes.',
		unknown: 'Not looked up yet. Press play and it will be resolved.'
	};

	const styles: Record<AudioKind, string> = {
		local: 'border-emerald-600/40 text-emerald-700 dark:text-emerald-400',
		preview: 'border-sky-600/40 text-sky-700 dark:text-sky-400',
		none: 'border-border text-muted-foreground',
		unknown: 'border-border text-muted-foreground'
	};
</script>

<span
	title={detail[kind]}
	class="inline-flex shrink-0 items-center gap-1 rounded border px-1.5 py-0.5 text-[10px] font-medium whitespace-nowrap {styles[
		kind
	]} {className}"
>
	{#if kind === 'local'}●{:else if kind === 'preview'}▷{:else}·{/if}
	{audioLabel(kind)}
</span>
