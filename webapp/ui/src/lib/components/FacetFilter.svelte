<script lang="ts">
	import type { FacetValue } from '$lib/api';

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

	<div class="max-h-56 space-y-0.5 overflow-y-auto pr-1">
		{#each filtered as value (value.value)}
			<label
				class="hover:bg-muted flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 text-xs"
			>
				<input
					type="checkbox"
					class="accent-primary size-3"
					checked={selected.includes(value.value)}
					onchange={() => toggle(value.value)}
				/>
				<span class="flex-1 truncate" title={value.value}>{value.value}</span>
				<span class="text-muted-foreground tabular-nums">{value.count}</span>
			</label>
		{/each}
		{#if !filtered.length}
			<p class="text-muted-foreground px-1.5 py-1 text-xs">no match</p>
		{/if}
	</div>

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
