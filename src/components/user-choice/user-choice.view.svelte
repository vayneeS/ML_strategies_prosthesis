<script>
  import { ViewContainer } from '@marcellejs/design-system';
  export let title;
  //export let pressed;
  //export let gesture;
  export let current = "pd";
  import { createEventDispatcher } from 'svelte';
  import {thumbnails,gifs} from './../../utils'

  const dispatch = createEventDispatcher();

  function forwardGesture(gesture) {
    dispatch('message', {
      data: gesture,
    })  
  }
  
//   const thumbnails = [
//   {name:'Palm Down', pic:'../images/wp.jpg'},
//   {name:'Palm Up', pic:'../images/ws.jpg'},
//   {name:'Close Hand', pic:'../images/hc.jpg'},
//   {name:'Open Hand', pic:'../images/ho.jpg'},
//   {name:'Close Pinch', pic:'../images/pc.png'},
//   {name:'Open Pinch', pic:'../images/po.png'},
//   {name:'Rest Hand', pic:'../images/rt.png'},
//   {name:'Point Index', pic:'../images/ip.jpg'},
// ];
  
</script>

<ViewContainer {title}>
  
  <div class = 'center-div'>
    <button class={current === 'pd' ? 'chosen' : 'choose'} on:click={() => (current = 'pd',forwardGesture('Palm Down'))}><img id='Palm Down' src={thumbnails['Palm Down']} alt='Palm Down' title='Palm Down' style="height: 50px; width: 50px"/></button>
    <button class={current === 'pu' ? 'chosen' : 'choose'} on:click={() => (current = 'pu',forwardGesture('Palm Up'))}><img id='Palm Up' src={thumbnails['Palm Up']} alt='Palm Up' title='Palm Up' style="height: 50px; width: 50px"/></button>
    <button class={current === 'ch' ? 'chosen' : 'choose'} on:click={() => (current = 'ch',forwardGesture('Close Hand'))}><img id='Close Hand' src={thumbnails['Close Hand']} alt='Close Hand' title='Close Hand' style="height: 50px; width: 50px"/></button>
    <button class={current === 'oh' ? 'chosen' : 'choose'} on:click={() => (current = 'oh',forwardGesture('Open Hand'))}><img id='Open Hand' src={thumbnails['Open Hand']} alt='Open Hand' title='Open Hand' style="height: 50px; width: 50px"/></button>
    <button class={current === 'cp' ? 'chosen' : 'choose'} on:click={() => (current = 'cp',forwardGesture('Close Pinch'))}><img id='Close Pinch' src={thumbnails['Close Pinch']} alt='Close Pinch' title='Close Pinch' style="height: 50px; width: 50px"/></button>
    <button class={current === 'op' ? 'chosen' : 'choose'} on:click={() => (current = 'op',forwardGesture('Open Pinch'))}><img id='Open Pinch' src={thumbnails['Open Pinch']} alt='Open Pinch' title='Open Pinch' style="height: 50px; width: 50px"/></button>
    <button class={current === 'rh' ? 'chosen' : 'choose'} on:click={() => (current = 'rh',forwardGesture('Rest Hand'))}><img id='Rest Hand' src={thumbnails['Rest Hand']} alt='Rest Hand' title='Rest Hand' style="height: 50px; width: 50px"/></button>
    <button class={current === 'pi' ? 'chosen' : 'choose'} on:click={() => (current = 'pi',forwardGesture('Point Index'))}><img id='Point Index' src={thumbnails['Point Index']} alt='Point Index' title='Point Index' style="height: 50px; width: 50px"/></button>
  </div>
  <!-- {#if clicked}
  {/if} -->
</ViewContainer>

<style>
  button {
    background-color: rgb(255, 255, 255);
    border-color: rgb(1, 1, 1);
    color: rgb(0, 0, 0);
    padding: 2px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 20px;
    cursor: pointer;
    height: 100px;
    width: 100px;
    overflow: hidden;
    border-radius: 50%;
    margin-top: 5px;
    margin-right: 5px;
    margin-left: 5px;
    margin-bottom: 5px;
  }
  .center-div {
    display: inline-block;
  } 
  .chosen {
    border-color: #ff3e00;
    background-color: rgb(255, 255, 255);
    border-style: solid;
    padding: 1px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 20px;
    cursor: pointer;
    height: 100px;
    width: 100px;
    overflow: hidden;
    border-radius: 50%;
    margin-top: 5px;
    margin-right: 5px;
    margin-left: 5px;
    margin-bottom: 5px;
  
  }
  .choose {
    background-color: rgb(255, 255, 255);
    border-color: rgb(0, 0, 0);
    border-style: solid;
    padding: 1px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 20px;
    cursor: pointer;
    height: 100px;
    width: 100px;
    overflow: hidden;
    border-radius: 50%;
    margin-top: 5px;
    margin-right: 5px;
    margin-left: 5px;
    margin-bottom: 5px;
  }
</style>
