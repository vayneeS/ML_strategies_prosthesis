import { Component, Stream } from '@marcellejs/core';
import View from './post-training.view.svelte';

export class PostTraining extends Component {
  constructor() {
    super();
    this.title = 'Gesture';
    
  }
  
  mount(target) {
    const t = target || document.querySelector(`#${this.id}`);
    if (!t) return;
    this.destroy();
    this.$$.app = new View({
      target: t,
      props: {
        title: this.title,
      },
    });
  }
}
