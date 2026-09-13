<script lang="ts">
	let { pathname = '/' }: { pathname?: string } = $props();
</script>

<header class="border-b">
	<div class="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
		<a href="/" class="flex items-center gap-2 font-semibold tracking-tight">
			<span class="bg-primary text-primary-foreground rounded px-1.5 py-0.5 text-xs">BT</span>
			MTG-Jamendo similarity explorer
		</a>
		<nav class="text-muted-foreground flex items-center gap-4 text-sm">
			<a
				href="/"
				class="hover:text-foreground transition-colors"
				class:text-foreground={pathname === '/'}
			>
				Browse
			</a>
			{#if pathname.startsWith('/track/')}
				<span class="text-foreground">Track</span>
			{/if}
		</nav>
		<p class="text-muted-foreground ml-auto hidden text-xs sm:block">
			32,783 tracks · 128-d Barlow Twins embeddings · exact cosine
		</p>
	</div>
</header>
