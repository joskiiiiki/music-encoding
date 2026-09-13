/**
 * One audio element for the whole app, so you can keep browsing a graph while a track
 * plays.
 *
 * Audio is the constrained part of this app: 586 of 32,783 tracks (~1.8%) exist locally
 * and roughly 28% of the rest resolve to a third-party 30s preview. So the player is
 * explicit about which of those it is playing, and about the two ways playback can fail
 * -- nothing available (404) versus the browser refusing the stream.
 *
 * A preview is also a *different recording* of the song, matched by fuzzy text search.
 * `matchedTitle`/`artistMatch` are surfaced for that reason, and `forgetMatch()` exists
 * so "not it?" does something rather than just hiding the label.
 */

import { api, type Track } from './api';

class Player {
	track = $state<Track | null>(null);
	playing = $state(false);
	loading = $state(false);
	currentTime = $state(0);
	duration = $state(0);
	error = $state<string | null>(null);
	/** What the server said it matched, when it differs from the corpus track. */
	matchedTitle = $state<string | null>(null);

	#audio: HTMLAudioElement | null = null;

	#element(): HTMLAudioElement {
		if (!this.#audio) {
			const audio = new Audio();
			audio.preload = 'none';
			audio.addEventListener('timeupdate', () => (this.currentTime = audio.currentTime));
			audio.addEventListener('durationchange', () =>
				(this.duration = Number.isFinite(audio.duration) ? audio.duration : (this.track?.duration ?? 0))
			);
			audio.addEventListener('play', () => (this.playing = true));
			audio.addEventListener('pause', () => (this.playing = false));
			audio.addEventListener('ended', () => (this.playing = false));
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

	async play(track: Track) {
		const audio = this.#element();
		this.error = null;
		this.matchedTitle = null;
		this.track = track;
		this.currentTime = 0;
		this.duration = track.duration ?? 0;
		this.loading = true;
		// A cache-busting query is deliberately absent: the API already re-resolves a
		// stale preview URL itself, and a stable URL lets the browser Range-cache it.
		audio.src = api.audioUrl(track.idx);
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
	}
}

export const player = new Player();
