import { Component,Stream } from '@marcellejs/core';
import View from './get-gesture.view.svelte';

export class GetGesture extends Component {
	constructor() {
		super();
		this.title = '';
		this.pressed = new Stream(false,false);
	}

	mount(target) {
		const t = target || document.querySelector(`#${this.id}`);
		if (!t) return;
		this.destroy();
		this.$$.app = new View({
			target: t,
			props: {
				title: this.title,
				pressed: this.$pressed,
			}
		});
		this.$$.app.$on('message', (e) => {
			if(e.detail.data){
				console.log('pressed',e.detail.data)
				this.pressed.set(true) 
				return this.pressed.value
			}
		})
	}
}
