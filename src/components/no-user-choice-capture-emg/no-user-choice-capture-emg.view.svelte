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
  import { createEventDispatcher } from 'svelte';
  const dispatch = createEventDispatcher();

  let clicked = false;

  function forwardMsg() {
    dispatch('message', {
      data: true,
    })//,
     clicked = true;
  }
  function startExp() {
    dispatch('message', {
      start: true,
    })//,
  }
  
  let done = false;

  function disable(e) {
    // get the button element and disable it
    const node = e.currentTarget;
    node.disabled = true;
  }
</script>

<ViewContainer {title}>
  <button id="btnStart" class = "start" on:click={(e) => {
    startExp()}}> Start phase 
  </button>
  <!-- <button id="btnStart" class = "start" on:click={(e) => {
    startExp(),disable(e)}}> Start Experiment 
  </button> -->
  {#if $completed !== undefined}
    <div class="completed">{$completed}</div>
  {/if}
  {#if $gesture !== undefined}
    <div class="gesture">{$gesture}</div>
  {/if}
  {#if $img !== undefined }
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
  
  <button 
    class={current === 'capture' ? 'startcapture' : 'nocapture'}
    on:click={() => {
        forwardMsg(),
        (current = 'capture'),
        setTimeout(function () {
          current = '';
        }, 2000)
    }}
  >
    Capture
  </button>
  
  <!-- {#if clickedGesture}
      <Counter on:completed="{() => {done = true,clicked=false}}" />
  {/if} -->
  
 <!-- </div> -->
</ViewContainer>

<style>
  button{
    width: 150px;
    margin-bottom: 1rem;
  }
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
</style>
