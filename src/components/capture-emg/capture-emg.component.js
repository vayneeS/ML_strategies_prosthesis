import { Component,Stream } from '@marcellejs/core';
import View from './capture-emg.view.svelte';

export class CaptureEmg extends Component {
	constructor() {
		super();
		this.current = ''
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
				current: this.current,
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
