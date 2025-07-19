import { Component } from '@marcellejs/core';
import View from './dataset.view.svelte';

export class Dataset extends Component {
	constructor(dataset) {
		super();
		this.title = '';
		this.dataset = dataset
	}

	mount(target) {
		const t = target || document.querySelector(`#${this.id}`);
		if (!t) return;
		this.destroy();
		this.$$.app = new View({
			target: t,
			props: {
				title: this.title,
				dataset: this.dataset
			}
		});
	}
}
