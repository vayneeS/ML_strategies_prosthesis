<script>
import { ViewContainer } from '@marcellejs/design-system';
export let title;
import { createEventDispatcher } from 'svelte';
import { tweened } from 'svelte/motion';
const dispatch = createEventDispatcher();
export let timer; 
// console.log(timer)
let count = tweened(timer)

  setInterval(() => {
      if ($count > 0) $count--;
  }, 1000);

  $: minutes = Math.floor($count / 60);
  $: minname = minutes > 1 ? "mins" : "min";
  $: seconds = Math.floor($count - minutes * 60)

  $: {
        if ($count === 0) {
            clearInterval(count);
            count = null;
            dispatch('completed');
        }
      }
</script>

<ViewContainer {title}>
  <h1>
    <!-- <span class="mins">{minutes}</span> -->
      start in : 
    <span class="secs">{seconds}</span>
  </h1>
</ViewContainer>

<style>
  .secs {
		color: rgb(249, 3, 3);  
        font-size: 30px;
	}
	h1{
        font-size: 30px;
    }
</style>
