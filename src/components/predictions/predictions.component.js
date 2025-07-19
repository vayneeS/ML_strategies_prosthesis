import { Component,Stream } from '@marcellejs/core';
import View from './predictions.view.svelte';
import {thumbnails,gifs} from './../../utils'

export class Predictions extends Component {
	constructor() {
		super();
		this.title = 'Prediction';
		this.$gesture = new Stream('', true);
		// this.thumbnails = {
		// 	'Palm Down': '../images/wp.jpg',
		// 	'Palm Up': '../images/ws.jpg',
		// 	'Close Hand': '../images/hc.jpg',
		// 	'Open Hand': '../images/ho.jpg',
		// 	'Close Pinch': '../images/pc.png',
		// 	'Open Pinch': '../images/po.png',
		// 	'Rest Hand': '../images/rt.png',
		// 	'Point Index': '../images/ip.jpg',
		//   };
		this.$pred = new Stream('', true);
	}
	async setPredImg(source) {
		if (source !== undefined && this.$pred !== undefined) {
		  this.$gesture.set(source)
			   
		  this.$pred.set(thumbnails[source]);
			  
		}

	  }
	mount(target) {
		const t = target || document.querySelector(`#${this.id}`);
		if (!t) return;
		this.destroy();
		this.$$.app = new View({
			target: t,
			props: {
				title: this.title,
				gesture: this.$gesture,
				pred: this.$pred
			}
		});
	}
}
