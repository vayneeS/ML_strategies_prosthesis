import { Component,Stream } from '@marcellejs/core';
import View from './no-user-choice-train-capture-emg.view.svelte';
import {thumbnails,gifs} from './../../utils'
export class NoUserChoiceTrainCaptureEmg extends Component {
	constructor() {
		super();
		this.pressed = new Stream(false,false);
		this.getGesture = new Stream(false,false);
		this.current = ''
		this.title = 'Gesture';
		this.$gesture = new Stream('', true);
		this.$completed = new Stream('', true);
		this.$link = new Stream('', true);
		this.$img = new Stream('', true);
		// this.$enableGetGesture = new Stream(false,false);
	}
	// async setLabel(source){
	// 	if (source !== undefined && this.$gesture !== undefined) {
	// 		this.$gesture.set(source)
	// 	}
	// }
	async setIMG(source) {

		if (source !== undefined && this.$gesture !== undefined) {
			this.$gesture.set(source)
			console.log(source)     
		//   this.$img.set(this.thumbnails[source]);
			this.$img.set(thumbnails[source]);		  
		}
		console.log(source)
		// if (source == '') {
		//   this.$img.set(undefined);
		//   this.$gesture.set('Completed!');
		// }

		}
		async setGIF(source) {
	
		if (source !== undefined && this.$gesture !== undefined) {
			this.$gesture.set(source)
				
		//   this.$link.set(this.gifs[source]);
			this.$link.set(gifs[source]);		  
		}
	
		if (source == '') {
			this.$link.set(undefined);
			this.$gesture.set('Completed!');
		}
		}
	
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
				getGesture: this.$getGesture,
				gesture: this.$gesture,
				completed:this.$completed,
				link: this.$link,
				img: this.$img,
				// enableGetGesture: this.$enableGetGesture
			}
		,
		});
		this.$$.app.$on('message', (e) => {
			if(e.detail.data){
				console.log('pressed',e.detail.data)
				this.pressed.set(true) ;
				return this.pressed.value;
			}
		})
		this.$$.app.$on('message', (e) => {
			if(e.detail.gest){
				this.getGesture.set(true); 
				return this.getGesture.value;
			}
		})
	}
}
