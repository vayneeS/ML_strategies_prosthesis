<script>
  import { ViewContainer } from '@marcellejs/design-system';
  export let title;
  export let dataset;
  
  $: count = dataset.$count;
  </script>
  
  <ViewContainer {title}>
    <p class = 'data-text'>
      {#if $count}
        Captured {$count} instance{$count ? 's' : ''}.
      {:else}
       No instances captured
      {/if}
    </p>
  </ViewContainer>
  
  <style>
   .data-text{
      font-size: 25px;
      color: black;
      text-align: center;
   }
  </style>
  