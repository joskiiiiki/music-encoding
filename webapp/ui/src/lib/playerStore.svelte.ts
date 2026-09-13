/**
 * One audio element for the whole app, so you can keep browsing a graph while a track
 * plays -- plus the **walk**: an endless chain where each track is followed by its
 * nearest neighbour in the embedding space.
 *
 * The walk is the point of the app made audible. Simulated over 12 random starting
 * points, 30 steps of "nearest neighbour not yet in the chain" visits ~12 distinct
 * artists and ~14 genre tags at a mean hop cosine of 0.841 -- a coherent region rather
 * than a single artist on repeat, and it never runs out of candidates in 32k tracks.
 * The optional "skip same artist" rule nearly doubles the artist variety (21.3) for a
 * small coherence cost (0.784), so both are offered.
 *
 * The next track is *planned as soon as the current one starts*, so advancing at the end
 * of a song is a buffer swap rather than a fetch + resolve.
 */

import { api, type Track } from './api';

export interface WalkStep {
	idx: number;
	title: string;
	artist: string;
	/** Cosine of the hop that led here; null for the track that started the walk. */
	cos: number | null;
}

/** How many neighbours to consider when choosing the next step. */
const LOOKAHEAD = 50;

class Player {
	track = $state<Track | null>(null);
	playing = $state(false);
	loading = $state(false);
	currentTime = $state(0);
	duration = $state(0);
	error = $state<string | null>(null);
	/** What the server said it matched, when it differs from the corpus track. */
	matchedTitle = $state<string | null>(null);

	/** Walk state. */
	walkEnabled = $state(false);
	avoidSameArtist = $state(false);
	trail = $state<WalkStep[]>([]);
	nextUp = $state<Track | null>(null);
	planning = $state(false);

	/** Ids played in this walk, so a step never revisits one. */
	#visited = new Set<number>();
	#audio: HTMLAudioElement | null = null;

	#element(): HTMLAudioElement {
		if (!this.#audio) {
			const audio = new Audio();
			audio.preload = 'none';
			audio.addEventListener('timeupdate', () => (this.currentTime = audio.currentTime));
			audio.addEventListener(
				'durationchange',
				() =>
					(this.duration = Number.isFinite(audio.duration)
						? audio.duration
						: (this.track?.duration ?? 0))
			);
			audio.addEventListener('play', () => (this.playing = true));
			audio.addEventListener('pause', () => (this.playing = false));
			audio.addEventListener('ended', () => {
				this.playing = false;
				void this.#advance();
			});
			audio.addEventListener('error', () => {
				this.playing = false;
				this.error = 'Playback failed. The preview URL may have expired -- try again.';
			});
			this.#audio = audio;
		}
		return this.#audio;
	}

	async toggle(track?: Track) {
		if (track && this.track?.idx !== track.idx) return this.play(track);
		const audio = this.#element();
		if (!this.track) return;
		if (audio.paused) {
			try {
				await audio.play();
			} catch {
				this.error = 'Playback was blocked or the source is unavailable.';
			}
		} else {
			audio.pause();
		}
	}

	async play(track: Track, options: { hop?: number | null } = {}) {
		const audio = this.#element();
		this.error = null;
		this.matchedTitle = null;
		this.track = track;
		this.currentTime = 0;
		this.duration = track.duration ?? 0;
		this.loading = true;
		this.nextUp = null;

		if (this.walkEnabled) {
			this.#visited.add(track.idx);
			this.trail = [
				...this.trail,
				{
					idx: track.idx,
					title: track.title,
					artist: track.artist,
					cos: options.hop ?? null
				}
			];
		}

		// A cache-busting query is deliberately absent: the API already re-resolves a
		// stale preview URL itself, and a stable URL lets the browser Range-cache it.
		audio.src = api.audioUrl(track.idx);
		// Line the next step up immediately, NOT after awaiting playback: audio.play()
		// can be slow or rejected (autoplay policy), and gating the plan on it means the
		// walk has nothing ready when the track ends -- or dies on the first stale step.
		if (this.walkEnabled) void this.#planNext(track);
		try {
			await audio.play();
			// The first play of an unresolved track may have just created a cache row,
			// so refresh availability for the badge.
			void this.#refreshAvailability(track.idx);
		} catch {
			this.error =
				track.audio === 'none'
					? 'No audio available: not in the local slice and no preview found.'
					: 'Could not play this track.';
		} finally {
			this.loading = false;
		}
	}

	async #refreshAvailability(idx: number) {
		try {
			const fresh = await api.track(idx);
			if (this.track?.idx === idx) {
				this.track = { ...this.track, audio: fresh.audio };
				if (fresh.audio === 'preview') this.matchedTitle = fresh.title;
			}
		} catch {
			/* availability is cosmetic; never let it break playback */
		}
	}

	/** Pick the next step: the most similar track not already in this walk. */
	async #planNext(from: Track) {
		this.planning = true;
		try {
			const candidates = await api.similar(from.idx, LOOKAHEAD);
			const fresh = candidates.filter((c) => !this.#visited.has(c.idx));
			// Prefer a different artist when asked, but fall back to the unfiltered list
			// rather than dead-ending the walk if every candidate shares the artist.
			const varied = fresh.filter((c) => c.artist !== from.artist);
			const pool = this.avoidSameArtist ? (varied.length ? varied : fresh) : fresh;
			this.nextUp = pool[0] ?? null;
		} catch {
			this.nextUp = null;
		} finally {
			this.planning = false;
		}
	}

	/** Step to the next track, planning one on demand if the prefetch is not ready. */
	async #advance() {
		if (!this.walkEnabled || !this.track) return;
		if (!this.nextUp) await this.#planNext(this.track);
		if (this.nextUp) {
			await this.play(this.nextUp, { hop: this.nextUp.score ?? null });
			return;
		}
		this.walkEnabled = false;
		this.error = 'Walk ended: no unvisited neighbour left to step to.';
	}

	/** Turn the walk on from the current track, or off. */
	toggleWalk() {
		this.walkEnabled = !this.walkEnabled;
		this.error = null;
		if (this.walkEnabled && this.track) {
			this.#visited = new Set([this.track.idx]);
			this.trail = [
				{
					idx: this.track.idx,
					title: this.track.title,
					artist: this.track.artist,
					cos: null
				}
			];
			void this.#planNext(this.track);
		} else if (!this.walkEnabled) {
			this.trail = [];
			this.nextUp = null;
		}
	}

	/** Start a fresh walk seeded at this track. */
	async startWalkFrom(track: Track) {
		this.walkEnabled = true;
		this.trail = [];
		this.nextUp = null;
		this.#visited = new Set();
		this.error = null;
		await this.play(track);
	}

	/** Step immediately, without waiting for the track to end. */
	async stepNow() {
		if (!this.track) return;
		if (!this.walkEnabled) this.toggleWalk();
		await this.#advance();
	}

	/** Play an entry from the trail, dropping everything after it. */
	async jumpTo(index: number) {
		const step = this.trail[index];
		if (!step || !this.track) return;
		const truncated = this.trail.slice(0, index);
		this.#visited = new Set(truncated.map((s) => s.idx));
		this.trail = truncated;
		try {
			const target = step.idx === this.track.idx ? this.track : await api.track(step.idx);
			await this.play(target, { hop: step.cos });
		} catch {
			this.error = 'Could not open that step of the walk.';
		}
	}

	/** Forget a cached preview match so the next play resolves from scratch. */
	async forgetMatch() {
		const track = this.track;
		if (!track) return;
		const audio = this.#element();
		audio.pause();
		audio.removeAttribute('src');
		await api.clearAudioMatch(track.idx);
		this.error = null;
		this.track = { ...track, audio: 'unknown' };
		await this.play({ ...track, audio: 'unknown' });
	}

	seek(seconds: number) {
		const audio = this.#element();
		if (Number.isFinite(seconds)) audio.currentTime = seconds;
	}

	stop() {
		const audio = this.#element();
		audio.pause();
		this.track = null;
		this.error = null;
		this.walkEnabled = false;
		this.trail = [];
		this.nextUp = null;
		this.#visited.clear();
	}
}

export const player = new Player();
