<script>
  import { ViewContainer } from '@marcellejs/design-system';
  export let pressed;
  export let start;
  export let current;
  export let link;
  export let img;
  export let title;
  export let gesture;
  export let completed;
  // export let enableGetGesture;
  export let cur = 'inactive';
  import { createEventDispatcher } from 'svelte';
  // import GestureDisplay from '../gesture-display/gesture-display.view.svelte';
  const dispatch = createEventDispatcher();

  let clickedCapture = false;
  // let clickedGetGesture = false;
  // let name = 'Get Gesture';

  function forwardPressed() {
    dispatch('message', {
      data: true,
    })//,
    //  clickedCapture = true;
    //  clickedGetGesture = false;
  }
  function getGesture() {
    dispatch('message', {
      gest: true,
    })//,
    clickedCapture = false;
    // clickedGetGesture = true;
  }

  function disable(e) {
    // get the button element and disable it
    const node = e.currentTarget;
    node.disabled = true;
  }
  // console.log('enabled: ',enableGetGesture.value);
  // let enable = enableGetGesture.value;
</script>

<ViewContainer {title}>
  <!-- <button id="btnStart" class = "start" on:click={(e) => {
    startExp()}}> Start phase 
  </button> -->
  <!-- <button id="btnStart" class = "start" on:click={(e) => {
    startExp(),disable(e)}}> Start Experiment 
  </button> -->
  {#if $completed !== undefined}
    <div class="completed">{$completed}</div>
  {/if}
  {#if $gesture !== undefined}
    <div class="gesture">{$gesture}</div>
  {/if}
  {#if $img !== undefined}
    <img
      src={$img}
      alt=""
      height="217.5"
      width="250"
      style= "display: block; margin: auto"
       />
    <!-- <img id="gesture" src={$img} alt="" style="height: 100px; margin: 5px auto" /> -->
  {/if}
  <!-- {#if $link !== undefined}
    <img
    src={$link}
    alt=""
    height="217.5"
    width="250"
    style= "display: block; margin: auto"
    />
  {/if} -->
  <!-- <div class= 'center-div'> -->
  <!-- <div class= 'button'>
    <button id="gesture"
      class={cur}
      on:click={() => {
        console.log('name',name)
        if(name === 'Get Gesture'){
          getGesture(),
          cur = 'active';
          current ='';
          name = 'Capture';
          
        }
        else if(name === 'Capture'){
          // setTimeout(function () {
          //   cur = '';
          // }, 2000)
          forwardPressed(),
          cur = 'inactive';
          current = 'capture';
          name = 'Get Gesture';
        }
      }} 
    >{name}
    </button>
  </div> -->
 <div class= 'button'>
  <button id="gesture"
    class={cur}
    on:click={() => {
        getGesture(),
        cur = 'active';
        current =''; 
      }
    } 
    > Get Gesture
  </button>
</div> 
<!-- disabled='{clickedGetGesture}' disabled='{enable}'-->

  <button 
    class={current === 'capture' ? 'startcapture' : 'nocapture'}
    on:click={(e) => {
        forwardPressed(),
        (current = 'capture'),
        setTimeout(function () {
          (current = '', clickedCapture = true); 
        }, 2000)
    }} disabled='{clickedCapture}'
  >
    Capture
  </button>
  <!-- disabled='{clickedCapture}' -->
  
  <!-- {#if clickedGesture}
      <Counter on:completed="{() => {done = true,clicked=false}}" />
  {/if} -->
  
 <!-- </div> -->
</ViewContainer>

<style>
  /* button{
    width: 150px;
    margin-bottom: 1rem;
  } */
  button.startcapture {
    background-color: #ff3e00;
    border: none;
    color: white;
    padding: 2px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 30px;
    cursor: pointer;
    height: 150px;
    width: 150px;
    border-radius: 50%;
    margin-top: 5%;
  }
  button.nocapture {
    background-color: green;
    border: none;
    color: white;
    padding: 2px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 30px;
    cursor: pointer;
    height: 150px;
    width: 150px;
    border-radius: 50%;
    margin-top: 5%;
  }
  button.inactive {
    background-color: rgb(218, 218, 218);
    border: none;
    color: rgba(0, 0, 0, 0.874);
    padding: 2px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 30px;
    cursor: pointer;
    height: 100px;
    width: 300px;
    border-radius: 5%;
    margin-top: 5%;
     
  }
  button.active {
    background-color: rgb(218, 218, 218);
    border: solid #000000;
    color: rgba(0, 0, 0, 0.874);
    padding: 2px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 30px;
    cursor: pointer;
    height: 100px;
    width: 300px;
    border-radius: 5%;
    margin-top: 5%;
  }
  div.comp{
    display: inline-block;
  }
  .gesture {
    font-size: 30px;
    color: black;
    text-align: center;
  }
  .completed {
    font-size: 40px;
    color: rgb(255, 2, 2);
    text-align: center;
  }
  /* .center-div {
     margin: 0 auto;
     width: 100px;
  }  */
  button:disabled,
  button[disabled]{
    border: 1px solid #999999;
    background-color: #cccccc;
    color: #666666;
  }
</style>
