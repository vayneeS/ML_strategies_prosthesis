<script>
  import { ViewContainer} from '@marcellejs/design-system';
  export let title;
  // export let clickedGesture;
  export let current;
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher();

  function forwardMsg() {
    dispatch('message', {
      data: true,
    })
    // clickedGesture = true;
  }
  
</script>

<ViewContainer {title}>
  <div class= 'button'>
    <button
      class={current === 'inactive' ? 'active' : ''}
      on:click={() => {
          forwardMsg(),
          (current = 'inactive'),
          setTimeout(function () {
            current = '';
          }, 2000)
      }}
    >Get gesture
    </button>
   </div>
</ViewContainer>

<style>
  button {
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
  }
  button.active {
    border: solid #000000;
  }
</style>
