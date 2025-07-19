import { Component,Stream } from '@marcellejs/core';
import View from './user-choice-capture-emg.view.svelte';

export class UserChoiceCaptureEmg extends Component {
	constructor() {
		super();
		this.title = '';
		this.pressed = new Stream(false,false);
		this.current = ''
		this.chosen = new Stream('',false);
		this.$completed = new Stream('', true);
	}

	// async setIMG(source) {

	// 	if (source !== undefined && this.$gesture !== undefined) {
	// 	  this.$gesture.set(source)
	// 	  // console.log(this.thumbnails[source])     
	// 	  this.$img.set(this.thumbnails[source]);
			  
	// 	}
	
	// 	if (source == '') {
	// 	  this.$img.set(undefined);
	// 	  this.$gesture.set('Completed!');
	// 	}
	//   }
	//   async setGIF(source) {
	
	// 	if (source !== undefined && this.$gesture !== undefined) {
	// 	  this.$gesture.set(source)
			   
	// 	  this.$link.set(this.gifs[source]);
			  
	// 	}
	
	// 	if (source == '') {
	// 	  this.$link.set(undefined);
	// 	  this.$gesture.set('Completed!');
	// 	}
	//   }
	mount(target) {
		console.log('mounted')
		const t = target || document.querySelector(`#${this.id}`);
		if (!t) return;
		this.destroy();
		this.$$.app = new View({
			target: t,
			props: {
				title: this.title,
				pressed: this.$pressed,
				current: this.current,
				//gesture: this.$gesture,
				// link: this.$link,
				// img: this.$img
				chosen: this.$chosen,
				completed:this.$completed,

			}
		,
		});
		this.$$.app.$on('message', (e) => {
			if(e.detail.pressed){
				console.log('pressed',e.detail.pressed)
				this.pressed.set(true) 
				return this.pressed.value
			}
			if(e.detail.data){
				console.log('gesture',e.detail.data)
				this.chosen.set(e.detail.data);
				return this.chosen.value
			}
		})
		// this.$$.app.$on('message', (e) => {
		// 	if(e.detail.data){
		// 		console.log('gesture',e.detail.data)
		// 		this.chosen.set(e.detail.data);
		// 		return this.chosen.value
		// 	}
		// })
	}
}
