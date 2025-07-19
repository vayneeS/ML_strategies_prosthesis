<script>
    import { createEventDispatcher } from 'svelte';
	import { tweened } from 'svelte/motion';
    const dispatch = createEventDispatcher();
    let original = 2; // TYPE NUMBER OF SECONDS HERE
	let timer = tweened(original)

//  import Typewriter from "svelte-typewriter";
    setInterval(() => {
        if ($timer > 0) $timer--;
    }, 1000);

    $: minutes = Math.floor($timer / 60);
    $: minname = minutes > 1 ? "mins" : "min";
    $: seconds = Math.floor($timer - minutes * 60)

    $: {
            if ($timer === 0) {
                clearInterval(timer);
                timer = null;
                dispatch('completed');
            }
        }
</script>

<main>
  <div class="flex">

   
  </div>

<h1>
	<!-- <span class="mins">{minutes}</span> -->
    Start in : 
	<span class="secs">{seconds}</span>
</h1>
 
  
</main>

<style>
    main {
        width: 600px;
        margin: 0 auto;
    }
	
	.mins {
		color: darkgoldenrod;
	}
	.secs {
		color: rgb(249, 3, 3);  
        font-size: 30px;
	}
	h1{
        font-size: 30px;
    }
  
</style>
