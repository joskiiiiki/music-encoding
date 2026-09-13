/**
 * Typed client for the FastAPI backend (see `webapp/api`).
 *
 * Requests go to `/api/...` on this origin and are proxied to the API by Vite (see
 * `vite.config.ts`), so there is no CORS and the `<audio>` element can use Range
 * requests against the same host it loaded the page from.
 *
 * The types mirror `webapp/api/app/schemas.py`; the response models there are the
 * source of truth, and `GET /openapi.json` is generated from them.
 */

export type AudioKind = 'local' | 'preview' | 'none' | 'unknown';
export type Space = 'whitened' | 'raw';

export interface Track {
	idx: number;
	track_id: string | null;
	track_num: number | null;
	title: string;
	artist: string;
	album: string;
	/** Alphabetically first genre tag -- not a primary genre. `tags` is the honest set. */
	genre: string | null;
	tags: string[];
	instrument: string | null;
	mood: string | null;
	released: number | null;
	duration: number | null;
	audio: AudioKind;
	/** Cosine to the query, when the track was returned by a ranked call. */
	score: number | null;
	degree: number | null;
	seed: boolean | null;
}

export interface FacetValue {
	value: string;
	count: number;
}

export interface Facets {
	genre: FacetValue[];
	instrument: FacetValue[];
	mood: FacetValue[];
}

export interface SearchResponse {
	total: number;
	limit: number;
	offset: number;
	sort: string;
	items: Track[];
	facets: Facets;
}

export interface TrackDetail extends Track {
	similar: Track[];
}

export interface Edge {
	source: number;
	target: number;
	weight: number;
}

export interface NeighbourAgreement {
	observed: number;
	chance: number;
	lift: number | null;
	compared: number;
	skipped: number;
	agreeing: number;
	seed_value: string | null;
	reason: string | null;
}

export interface TagOverlap {
	mean_overlap: number;
	chance: number;
	lift: number | null;
	compared: number;
	seed_tags: string[];
	reason: string | null;
}

export interface Enrichment {
	artist: NeighbourAgreement;
	genre: NeighbourAgreement;
	tags: TagOverlap;
}

export interface GraphResponse {
	seed: number;
	space: string;
	nodes: Track[];
	edges: Edge[];
	enrichment: Enrichment;
}

export interface AudioStats {
	total_tracks: number;
	local: number;
	deezer: number;
	itunes: number;
	playable: number;
	attempted_misses: number;
	unresolved: number;
	attempted: number;
}

export interface Stats {
	tracks: number;
	dim: number;
	spaces: string[];
	meta: Record<string, string>;
	audio: AudioStats;
	facets: Record<string, number>;
}

type Params = Record<string, string | number | boolean | string[] | null | undefined>;

async function get<T>(path: string, params: Params = {}): Promise<T> {
	const url = new URL(path, window.location.origin);
	for (const [key, value] of Object.entries(params)) {
		if (value === null || value === undefined || value === '') continue;
		if (Array.isArray(value)) {
			// Repeated params: genre=a&genre=b, matching FastAPI's list[str] Query.
			for (const item of value) url.searchParams.append(key, item);
		} else {
			url.searchParams.set(key, String(value));
		}
	}
	const response = await fetch(url);
	if (!response.ok) {
		const detail = await response.text().catch(() => '');
		throw new Error(`${response.status} ${response.statusText}${detail ? `: ${detail}` : ''}`);
	}
	return (await response.json()) as T;
}

export interface SearchParams {
	q?: string;
	genre?: string[];
	instrument?: string[];
	mood?: string[];
	year_min?: number;
	year_max?: number;
	playable_only?: boolean;
	sort?: string;
	limit?: number;
	offset?: number;
}

export const api = {
	search: (params: SearchParams) => get<SearchResponse>('/api/search', params as Params),
	facets: () => get<Facets>('/api/facets'),
	track: (idx: number) => get<TrackDetail>(`/api/tracks/${idx}`),
	similar: (idx: number, k = 20, space: Space = 'whitened') =>
		get<Track[]>(`/api/tracks/${idx}/similar`, { k, space }),
	graph: (seed: number, k = 20, space: Space = 'whitened', minSim = 0) =>
		get<GraphResponse>('/api/graph', { seed, k, space, min_sim: minSim }),
	stats: () => get<Stats>('/api/stats'),

	/** Same-origin URL for the audio element; the API streams or proxies from here. */
	audioUrl: (idx: number) => `/api/tracks/${idx}/audio`,

	/**
	 * Forget a cached preview match. Preview matching is fuzzy, so when the player
	 * lands on a different recording the user needs a way to say so.
	 */
	clearAudioMatch: async (idx: number) => {
		await fetch(`/api/tracks/${idx}/audio`, { method: 'DELETE' });
	}
};

export function formatDuration(seconds: number | null): string {
	if (!seconds) return '--:--';
	const total = Math.round(seconds);
	const minutes = Math.floor(total / 60);
	return `${minutes}:${String(total % 60).padStart(2, '0')}`;
}

export function audioLabel(kind: AudioKind): string {
	switch (kind) {
		case 'local':
			return 'full track';
		case 'preview':
			return '30s preview';
		case 'none':
			return 'no audio';
		default:
			return 'not resolved';
	}
}
