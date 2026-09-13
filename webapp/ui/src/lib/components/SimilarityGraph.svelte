<script lang="ts">
	import { onMount } from 'svelte';
	import {
		forceCenter,
		forceCollide,
		forceLink,
		forceManyBody,
		forceSimulation,
		type Simulation,
		type SimulationLinkDatum,
		type SimulationNodeDatum
	} from 'd3-force';
	import { drag } from 'd3-drag';
	import { select } from 'd3-selection';
	import { zoom, zoomIdentity, type ZoomTransform } from 'd3-zoom';
	import { scaleOrdinal } from 'd3-scale';
	import { schemeTableau10 } from 'd3-scale-chromatic';
	import { formatDuration, type GraphResponse, type Track } from '$lib/api';

	type Node = Track & SimulationNodeDatum & { radius: number; color: string; label: string };
	type Link = SimulationLinkDatum<Node> & { weight: number };

	let {
		graph,
		minSim = 0,
		colorBy = 'genre',
		onReseed = undefined,
		height = 520
	}: {
		graph: GraphResponse;
		minSim?: number;
		colorBy?: 'genre' | 'artist';
		onReseed?: (idx: number) => void;
		height?: number;
	} = $props();

	let canvas: HTMLCanvasElement;
	let wrapper: HTMLDivElement;
	let hovered = $state<Node | null>(null);
	let legend = $state<{ label: string; color: string }[]>([]);
	// Edge count is mirrored into $state for the caption; `nodes`/`links` stay plain
	// arrays because d3-force mutates node positions every tick and proxying those
	// writes would be pointless overhead.
	let edgeCount = $state(0);
	let nodes: Node[] = [];
	let links: Link[] = [];
	let simulation: Simulation<Node, Link> | undefined;
	let transform: ZoomTransform = zoomIdentity;
	let theme = { bg: '#fff', edge: '#999', label: '#333', halo: '#fff', ring: '#111' };
	let width = $state(900);

	const palette = schemeTableau10;

	function labelOf(node: Track): string {
		if (colorBy === 'artist') return node.artist || 'unknown artist';
		return node.tags.length ? node.tags[0] : (node.genre ?? 'untagged');
	}

	function readTheme() {
		const style = getComputedStyle(document.documentElement);
		const read = (name: string, fallback: string) =>
			style.getPropertyValue(name).trim() || fallback;
		theme = {
			bg: read('--graph-bg', '#fff'),
			edge: read('--graph-edge', '#999'),
			label: read('--graph-label', '#333'),
			halo: read('--graph-halo', '#fff'),
			ring: read('--graph-seed-ring', '#111')
		};
	}

	function build() {
		// `.range(palette)` is load-bearing: without it the ordinal scale returns
		// undefined for every input, an invalid fillStyle is silently ignored by the
		// canvas, and every node is drawn black.
		const color = scaleOrdinal<string, string>()
			.domain([...new Set(graph.nodes.map(labelOf))].sort())
			.range(palette);
		// The seed is pinned at the centre; the rest start on a circle in index order so
		// the layout is reproducible for a given graph rather than depending on force-
		// solver tie-breaking.
		const radius = 170;
		nodes = graph.nodes.map((node, index) => {
			const degree = node.degree ?? 0;
			const spread = index === 0 ? 0 : index / Math.max(1, graph.nodes.length - 1);
			const angle = spread * Math.PI * 2;
			return {
				...node,
				// Small dots: structure and labels should carry the picture, not blobs.
				// Was 5 + min(9, ...) = 5..14; now roughly 1.8..4.5 (about a third of the
				// radius, a tenth of the area).
				radius: 1.8 + Math.min(2.7, Math.sqrt(degree) * 0.7),
				color: color(labelOf(node)),
				label: labelOf(node),
				x: index === 0 ? 0 : Math.cos(angle) * radius,
				y: index === 0 ? 0 : Math.sin(angle) * radius,
				...(index === 0 ? { fx: 0, fy: 0 } : {})
			};
		});
		const byIdx = new Map(nodes.map((node) => [node.idx, node]));
		links = graph.edges
			.filter((edge) => edge.weight >= minSim)
			.map((edge) => ({
				source: byIdx.get(edge.source)!,
				target: byIdx.get(edge.target)!,
				weight: edge.weight
			}))
			.filter((link) => link.source && link.target);

		const counts = new Map<string, number>();
		for (const node of nodes) counts.set(node.label, (counts.get(node.label) ?? 0) + 1);
		legend = [...counts.entries()]
			.sort((a, b) => b[1] - a[1])
			.slice(0, 8)
			.map(([label]) => ({ label, color: color(label) }));
		edgeCount = links.length;
	}

	function draw() {
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const dpr = window.devicePixelRatio || 1;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.clearRect(0, 0, width, height);
		ctx.save();
		ctx.translate(width / 2 + transform.x, height / 2 + transform.y);
		ctx.scale(transform.k, transform.k);

		// Edges first, so nodes sit on top. Width and alpha both track the cosine: the
		// graph is close to complete at low cosines, so weight is what carries meaning.
		for (const link of links) {
			const source = link.source as Node;
			const target = link.target as Node;
			const t = Math.max(0, Math.min(1, (link.weight - minSim) / Math.max(1e-6, 1 - minSim)));
			ctx.strokeStyle = theme.edge;
			ctx.globalAlpha = 0.1 + 0.55 * t;
			ctx.lineWidth = (0.5 + 2.5 * t) / transform.k;
			ctx.beginPath();
			ctx.moveTo(source.x!, source.y!);
			ctx.lineTo(target.x!, target.y!);
			ctx.stroke();
		}
		ctx.globalAlpha = 1;

		const labelled = new Set(
			nodes
				.slice()
				.sort((a, b) => (b.degree ?? 0) - (a.degree ?? 0))
				.slice(0, 8)
				.map((node) => node.idx)
		);
		labelled.add(graph.seed);

		for (const node of nodes) {
			ctx.beginPath();
			ctx.arc(node.x!, node.y!, node.radius, 0, Math.PI * 2);
			ctx.fillStyle = node.color;
			ctx.fill();
			if (node.seed) {
				// The seed is the thing you navigate by, so it gets a ring rather than
				// just being another dot.
				ctx.lineWidth = 1.5 / transform.k;
				ctx.strokeStyle = theme.ring;
				ctx.stroke();
			}
			if (hovered?.idx === node.idx) {
				ctx.lineWidth = 2 / transform.k;
				ctx.strokeStyle = theme.label;
				ctx.stroke();
			}
		}

		ctx.font = `${11 / transform.k}px ui-sans-serif, system-ui, sans-serif`;
		ctx.textAlign = 'center';
		ctx.textBaseline = 'top';
		ctx.lineWidth = 3 / transform.k;
		for (const node of nodes) {
			if (!labelled.has(node.idx)) continue;
			const text = node.title.length > 26 ? `${node.title.slice(0, 25)}…` : node.title;
			const y = node.y! + node.radius + 3;
			ctx.strokeStyle = theme.halo;
			ctx.strokeText(text, node.x!, y);
			ctx.fillStyle = theme.label;
			ctx.fillText(text, node.x!, y);
		}
		ctx.restore();
	}

	function restart() {
		simulation?.stop();
		build();
		simulation = forceSimulation(nodes)
			.force(
				'link',
				forceLink<Node, Link>(links)
					.id((node) => node.idx)
					// Weight drives BOTH strength and distance: without this the graph is
					// a hairball, because at these cosines nearly every pair is an edge.
					.strength((link) => Math.pow(link.weight, 4) * 1.2)
					.distance((link) => 30 + 190 * (1 - link.weight))
			)
			.force('charge', forceManyBody<Node>().strength(-190))
			.force('collide', forceCollide<Node>().radius((node) => node.radius + 3))
			.force('center', forceCenter(0, 0))
			.on('tick', draw);
		simulation.tick(120);
		// Stop after settling. Without this the simulation's own timer keeps ticking (and
		// redrawing) for a couple of seconds after the layout is already settled, and any
		// stray alphaTarget left by a drag that ended off-canvas keeps it hot forever --
		// which reads as the dots twitching.
		simulation.stop();
		draw();
	}

	onMount(() => {
		readTheme();
		const observer = new ResizeObserver((entries) => {
			width = Math.max(320, entries[0].contentRect.width);
			resizeCanvas();
			draw();
		});
		observer.observe(wrapper);

		const behaviour = zoom<HTMLCanvasElement, unknown>()
			.scaleExtent([0.3, 5])
			.on('zoom', (event) => {
				transform = event.transform;
				draw();
			});
		select(canvas).call(behaviour);

		select<HTMLCanvasElement, unknown>(canvas).call(
			drag<HTMLCanvasElement, unknown>()
				.subject((event) => {
					const node = findNode(event.x, event.y);
					return node ?? { x: 0, y: 0 };
				})
				.on('start', (event) => {
					if (!event.active) simulation?.alphaTarget(0.25).restart();
					const node = event.subject as Node;
					node.fx = node.x;
					node.fy = node.y;
				})
				.on('drag', (event) => {
					const node = event.subject as Node;
					if (!nodes.includes(node)) return;
					node.fx = toLocalX(event.x);
					node.fy = toLocalY(event.y);
				})
				.on('end', (event) => {
					if (!event.active) simulation?.alphaTarget(0);
					const node = event.subject as Node;
					if (node && nodes.includes(node) && !node.seed) {
						// Released nodes stay put (pinned) so a hand-arranged layout survives.
						node.fx = node.x;
						node.fy = node.y;
					}
				})
		);

		const media = window.matchMedia('(prefers-color-scheme: dark)');
		const onTheme = () => {
			readTheme();
			draw();
		};
		media.addEventListener('change', onTheme);

		// A drag released outside the canvas never fires the drag behaviour's `end`, which
		// would leave alphaTarget at 0.25 and the simulation heating forever.
		const releaseDrag = () => simulation?.alphaTarget(0);
		window.addEventListener('pointerup', releaseDrag);

		return () => {
			observer.disconnect();
			media.removeEventListener('change', onTheme);
			window.removeEventListener('pointerup', releaseDrag);
			simulation?.stop();
		};
	});

	function toLocalX(screenX: number) {
		const rect = canvas.getBoundingClientRect();
		return (screenX - rect.width / 2 - transform.x) / transform.k;
	}
	function toLocalY(screenY: number) {
		const rect = canvas.getBoundingClientRect();
		return (screenY - rect.height / 2 - transform.y) / transform.k;
	}
	function findNode(screenX: number, screenY: number): Node | null {
		const x = toLocalX(screenX);
		const y = toLocalY(screenY);
		return (
			// Generous minimum hit radius: the dots are small now, but they still have to be
			// easy to hover and to click to recentre.
			nodes.find(
				(node) =>
					Math.hypot(node.x! - x, node.y! - y) <= Math.max(node.radius + 4, 8)
			) ?? null
		);
	}

	function resizeCanvas() {
		if (!canvas) return;
		const dpr = window.devicePixelRatio || 1;
		canvas.width = Math.round(width * dpr);
		canvas.height = Math.round(height * dpr);
		canvas.style.width = `${width}px`;
		canvas.style.height = `${height}px`;
	}

	// Rebuild only when the graph or a display parameter actually changes. Keyed on a
	// signature rather than on the `graph` object's identity: restart() re-seeds every
	// node onto the starting circle, so letting it run on an unrelated re-render makes the
	// whole layout jump.
	let lastSignature = '';
	$effect(() => {
		const signature = `${graph.seed}|${graph.nodes.length}|${graph.edges.length}|${minSim}|${colorBy}`;
		if (signature === lastSignature) return;
		lastSignature = signature;
		restart();
	});

	$effect(() => {
		// Redraw when the hovered node changes. The canvas is otherwise only painted on a
		// simulation tick, and the simulation now stops once settled -- so without this the
		// hover highlight would never appear.
		void hovered;
		draw();
	});
</script>

<div class="relative" bind:this={wrapper}>
	<!-- Marked aria-hidden: this is a picture of the same information the ranked
	     neighbour list beside it already provides in text, so exposing a canvas with no
	     accessible structure would add noise, not access. -->
	<canvas
		bind:this={canvas}
		class="border-border w-full cursor-grab touch-none rounded-lg border active:cursor-grabbing"
		style="background: var(--graph-bg)"
		aria-hidden="true"
		onpointermove={(event) => {
			const rect = canvas.getBoundingClientRect();
			hovered = findNode(event.clientX - rect.left, event.clientY - rect.top);
		}}
		onpointerleave={() => (hovered = null)}
		onclick={() => {
			if (hovered && onReseed && !hovered.seed) {
				onReseed(hovered.idx);
				hovered = null;
			}
		}}
	></canvas>

	{#if hovered}
		<div
			class="bg-popover text-popover-foreground border-border pointer-events-none absolute z-10 max-w-xs rounded-lg border px-2.5 py-1.5 shadow-sm"
			style="left: 8px; top: 8px"
		>
			<p class="text-xs font-medium">{hovered.title}</p>
			<p class="text-muted-foreground text-xs">{hovered.artist}</p>
			<p class="text-muted-foreground text-xs">
				{hovered.tags.slice(0, 3).join(', ') || 'no tags'} · {hovered.degree} edges
				{#if hovered.score !== null}&nbsp;· cos {hovered.score.toFixed(3)}{/if}
			</p>
			{#if !hovered.seed && onReseed}
				<p class="text-muted-foreground mt-0.5 text-[10px]">click to recentre</p>
			{/if}
		</div>
	{/if}

	<div class="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1">
		{#each legend as item (item.label)}
			<span class="text-muted-foreground flex items-center gap-1 text-[10px]">
				<span class="size-2 rounded-full" style="background: {item.color}"></span>
				{item.label}
			</span>
		{/each}
		<span class="text-muted-foreground ml-auto text-[10px]">
			colour = {colorBy} · size = degree · {edgeCount} edges · drag to move, scroll to zoom,
			click to recentre
		</span>
	</div>
</div>
