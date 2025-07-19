import { Component, Stream } from '@marcellejs/core';
import View from './experiment.view.svelte';

export class Experiment extends Component {
//   state = new Stream('',false);

//   // state machine
//   experiment = Machine<CounterContext>(trial_state, trial_action(this));
 
//   service = interpret(this.experiment, { devTools: true });

//   push_start() {
//     if (this.service.state.matches('init')) {
//       this.service.send('START');
//       this.state.set(this.service.state.value.toString());
//       console.log("PUSH START");
//     }
//   }

//   constructor(datastore
//               ) {
//     super();
//     this.datastore = datastore;
 
//     // this.ws = new WebSocket('ws://127.0.0.1:8765/');
//     // this.ws.onerror = (e: any) => {
//     //   e.name = 'Websocket connection error';
//     //   e.message = `Connection failed with websocket server ${e.target.url}`;
//     // };


//     this.ws.onmessage = (event) => {
//       const { type, data } = JSON.parse(event.data);
//       if (type == 'icf') {
//         console.log("icf: ", data);
//         this.icf = data;

//         // this takes care of the first value: undefined
//         var past_value = this.$icf.value;
//         if (past_value == undefined) {
//           past_value = [{x: this.service.state.context.trial_id, y: data}];
//         }
//         else {
//           past_value.push({x: this.service.state.context.trial_id, y: data});
//         }

//         this.$icf.set(past_value);
//         this.service.send('NEXT');
//       }

//       if (type == 'arm') {
//         console.log("chosen arm: ", data);

        
//         this.service.send('NEXT');
//       }


//     }

//     // // debug transitions
//     // this.service.onTransition(state => {console.log(state.value);});
//     this.service.start();
//     this.start()
//   }

//   mount(target) {
//     const target = document.querySelector(targetSelector || `#${this.id}`);
//     if (!target) return;
//     this.destroy();
//     this.$$.app = new Component({
//       target,
//       props: {
//         title: this.name,
//         state: this.state,

//         user_id: this.user_id,
//       },
//     });



//     // DISPATCH callback
//     this.$$.app.$on('message', (e) => {
//       if (e.detail.cmd == 'start') {

//         this.service.state.context.width = this.$width.value;
//         this.service.state.context.length = this.$length.value;
//         this.service.state.context.block_length = this.$block_length.value;
//         this.service.state.context.mouvement_time = this.$mouvement_time.value;

//         this.push_start();
//       }
//       if (e.detail.cmd == 'set_optitrack_origin') {
//         this.ws.send(JSON.stringify({ action: 'set_optitrack_origin'}));
//       }
//     });
//     this.$$.app.$on('update', (e) => {
//       // updates the parameters from UI
//       if (e.detail.type == 'width') {
//         this.$width.set(parseInt(e.detail.value));
//       }
//       if (e.detail.type == 'length') {
//         this.$length.set(parseInt(e.detail.value));
//       }
//       if (e.detail.type == 'block_length') {
//         this.$block_length.set(parseInt(e.detail.value));
//       }
//       if (e.detail.type == 'mouvement_time') {
//         this.$mouvement_time.set(parseInt(e.detail.value));
//       }

//     });

//   }
}
