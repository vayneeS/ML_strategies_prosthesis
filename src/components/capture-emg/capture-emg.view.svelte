<script>
  import { ViewContainer } from '@marcellejs/design-system';
    export let pressed;
    export let current;
    export let title;
    import { createEventDispatcher } from 'svelte';
    const dispatch = createEventDispatcher();
  
    let clicked = false;
  
    function forwardMsg() {
      dispatch('message', {
        data: true,
      })//,
       clicked = true;
    }
  </script>
  <ViewContainer {title}>
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
  
    /* .center-div {
       margin: 0 auto;
       width: 100px;
    }  */
  </style>
  
  