<script lang="ts">
	import { audioLabel, type AudioKind } from '$lib/api';
	import { Badge } from '$lib/components/ui/badge/index.js';
	import * as Tooltip from '$lib/components/ui/tooltip/index.js';
	import { cn } from '$lib/utils.js';

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
		none: 'text-muted-foreground',
		unknown: 'text-muted-foreground'
	};

	const glyph: Record<AudioKind, string> = {
		local: '●',
		preview: '▷',
		none: '·',
		unknown: '·'
	};
</script>

<Tooltip.Root>
	<Tooltip.Trigger>
		{#snippet child({ props })}
			<Badge
				variant="outline"
				{...props}
				class={cn('h-5 gap-1 rounded px-1.5 text-[10px]', styles[kind], className)}
			>
				{glyph[kind]}
				{audioLabel(kind)}
			</Badge>
		{/snippet}
	</Tooltip.Trigger>
	<Tooltip.Content>
		<p class="max-w-xs text-xs">{detail[kind]}</p>
	</Tooltip.Content>
</Tooltip.Root>
