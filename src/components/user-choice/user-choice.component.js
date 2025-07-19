import { Component,Stream } from '@marcellejs/core';
import View from './user-choice.view.svelte';

export class UserChoice extends Component {
	constructor() {
		super();
		this.title = 'Choose gesture';
		this.current = new Stream('',false);
	}  

	mount(target) {
		const t = target || document.querySelector(`#${this.id}`);
		if (!t) return;
		this.destroy();
		this.$$.app = new View({
			target: t,
			props: {
				title: this.title,
				current: this.$current,
				// chosen: this.$chosen,
			}
		});
		this.$$.app.$on('message', (e) => {
			if(e.detail.data){
				console.log('pressed',e.detail.data)
				this.current.set(e.detail.data);
				return this.current.value
			}
		})
	}
}
