<script lang="ts">
	import type { FacetValue } from '$lib/api';
	import { Checkbox } from '$lib/components/ui/checkbox/index.js';
	import { ScrollArea } from '$lib/components/ui/scroll-area/index.js';

	let {
		title,
		values,
		selected,
		limit = 12,
		onselect
	}: {
		title: string;
		values: FacetValue[];
		selected: string[];
		limit?: number;
		onselect: (next: string[]) => void;
	} = $props();

	// Counts come from the server with this facet's own filter removed, so an unselected
	// option shows how many tracks selecting it would add rather than reading zero.
	let query = $state('');
	let expanded = $state(false);

	const filtered = $derived(
		query
			? values.filter((value) => value.value.includes(query.toLowerCase()))
			: expanded
				? values
				: values.slice(0, limit)
	);

	// An id has to be whitespace-free, and tag values are data rather than slugs.
	const domId = (value: string) => `facet-${title}-${value.replace(/[^a-z0-9]+/gi, '-')}`;

	function toggle(value: string) {
		onselect(
			selected.includes(value) ? selected.filter((item) => item !== value) : [...selected, value]
		);
	}
</script>

<div class="border-b py-3">
	<div class="mb-2 flex items-baseline justify-between">
		<h3 class="text-xs font-semibold tracking-wide uppercase">{title}</h3>
		{#if selected.length}
			<button
				type="button"
				class="text-muted-foreground hover:text-foreground text-[10px] underline"
				onclick={() => onselect([])}
			>
				clear
			</button>
		{/if}
	</div>

	{#if values.length > limit}
		<input
			type="text"
			bind:value={query}
			placeholder="filter {title}…"
			class="border-input mb-2 w-full rounded border px-2 py-1 text-xs outline-none"
		/>
	{/if}

	<!-- type="always": bits-ui defaults to "hover", which mounts the scrollbar only while
	     the pointer is over the area -- in a filter rail that reads as a list cut off with
	     no indication there is more of it. -->
	<ScrollArea class="h-56" type="always">
		<div class="space-y-0.5 pr-3">
			{#each filtered as value (value.value)}
				<div class="hover:bg-muted flex items-center gap-2 rounded px-1.5 py-1">
					<!-- The label is a sibling pointing at the checkbox by id, not a wrapper:
					     bits-ui renders the checkbox as a <button>, and a label wrapping a
					     button does not forward clicks to it. -->
					<Checkbox
						id={domId(value.value)}
						checked={selected.includes(value.value)}
						onCheckedChange={() => toggle(value.value)}
					/>
					<label
						for={domId(value.value)}
						class="flex-1 cursor-pointer truncate text-xs"
						title={value.value}
					>
						{value.value}
					</label>
					<span class="text-muted-foreground text-xs tabular-nums">{value.count}</span>
				</div>
			{/each}
			{#if !filtered.length}
				<p class="text-muted-foreground px-1.5 py-1 text-xs">no match</p>
			{/if}
		</div>
	</ScrollArea>

	{#if !query && values.length > limit}
		<button
			type="button"
			class="text-muted-foreground hover:text-foreground mt-1 text-[10px] underline"
			onclick={() => (expanded = !expanded)}
		>
			{expanded ? 'show fewer' : `show all ${values.length}`}
		</button>
	{/if}
</div>
