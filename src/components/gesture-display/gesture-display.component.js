import { Component, Stream } from '@marcellejs/core';
import View from './gesture-display.view.svelte';
import {thumbnails,gifs} from './../../utils'

export class GestureDisplay extends Component {
  constructor() {
    super();
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
  async setIMG(source) {

    if (source !== undefined && this.$gesture !== undefined) {
      this.$gesture.set(source)
      // console.log(this.thumbnails[source])     
      this.$img.set(thumbnails[source]);
          
    }
    if (source == '') {
      //this.$img.set(undefined);
      this.$completed.set('Completed!');
    }
  }
  async setGIF(source) {

    if (source !== undefined && this.$gesture !== undefined) {
      this.$gesture.set(source)
           
      this.$link.set(gifs[source]);
          
    }

    if (source == '') {
      this.$link.set(undefined);
      this.$gesture.set('Completed!');
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
        completed:this.$completed,
        link: this.$link,
        img: this.$img
      },
    });
  }
}
