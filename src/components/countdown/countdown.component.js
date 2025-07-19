import { Component } from '@marcellejs/core';
import View from './countdown.view.svelte';

export class Countdown extends Component {
	constructor(val) {
		super();
		this.title = '';
		this.timer = val;
	}

	mount(target) {
		const t = target || document.querySelector(`#${this.id}`);
		if (!t) return;
		this.destroy();
		this.$$.app = new View({
			target: t,
			props: {
				title: this.title,
				timer: this.timer
			}
		});
	}
}
