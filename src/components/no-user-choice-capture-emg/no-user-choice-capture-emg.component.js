import { Component,Stream } from '@marcellejs/core';
import View from './no-user-choice-capture-emg.view.svelte';
import {thumbnails,thumbnail_pos_neg,gifs} from './../../utils'

export class NoUserChoiceCaptureEMG extends Component {
	constructor() {
		super();
		this.pressed = new Stream(false,false);
		this.start = new Stream(false,false);
		this.current = ''
		this.title = 'Gesture';
		this.$gesture = new Stream('', true);
		this.$completed = new Stream('', true);
		this.$link = new Stream('', true);
		this.$img = new Stream('', true);

    // this.gifs = {
    //   'Palm Down': '../images/wp.gif',
    //   'Palm Up': '../images/ws.gif',
    //   'Close Hand': '../images/hc.gif',
    //   'Open Hand': '../images/ho.gif',
    //   'Close Pinch': '../images/pc.gif',
    //   'Open Pinch': '../images/po.gif',
    //   'Rest Hand': '../images/rt.gif',
    //   'Point Index': '../images/ip.gif',
    // };

    // this.thumbnails = {
    //   'Palm Down': '../images/wp.jpg',
    //   'Palm Up': '../images/ws.jpg',
    //   'Close Hand': '../images/hc.jpg',
    //   'Open Hand': '../images/ho.jpg',
    //   'Close Pinch': '../images/pc.png',
    //   'Open Pinch': '../images/po.png',
    //   'Rest Hand': '../images/rt.png',
    //   'Point Index': '../images/ip.jpg',
    // };
	}
	async setLabel(source){
		if (source !== undefined && this.$gesture !== undefined) {
			this.$gesture.set(source);
		}
	}
	async setIMG(source) {

		if (source !== undefined && this.$gesture !== undefined) {
		  this.$gesture.set(source)
		  console.log(source)     
		//   this.$img.set(thumbnails[source]);
		  		  
		}
		if(source.includes('Positive') || source.includes('Negative')){
			this.$img.set(thumbnail_pos_neg[source])
		}
		else{
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
				start: this.start,
				gesture: this.$gesture,
				completed:this.$completed,
				link: this.$link,
				img: this.$img,
			}
		,
		});
		this.$$.app.$on('message', (e) => {
			if(e.detail.data){
				console.log('pressed',e.detail.data)
				this.pressed.set(true) 
				return this.pressed.value
			}
		})
		this.$$.app.$on('message', (e) => {
			if(e.detail.start){
				this.start.set(true) 
				return this.start.value
			}
		})
	}
}
