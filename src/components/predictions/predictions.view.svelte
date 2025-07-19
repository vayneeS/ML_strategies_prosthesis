<script>
  import { ViewContainer } from '@marcellejs/design-system';
  export let pred;
  export let title;
  export let gesture = '';
  const obj = {'':-1,'Palm Up':0,'Palm Down':1,'Close Hand':2,'Open Hand':3,'Close Pinch':4,'Open Pinch':5,'Rest Hand':6,'Point Index':7};
  </script>
  
  <ViewContainer {title}>
    {#if $gesture !== undefined}
      <div class="gesture">{obj[$gesture]}</div>
    {/if}
    <!-- {#if $pred !== undefined}
      <img id="gesture" src={$pred} alt="" style="height: 100px; margin: 5px auto" />
    {/if} -->
  </ViewContainer>
  
  <style>
    .gesture {
      font-size: 20px;
      color: black;
      text-align: center;
    }
  </style>
  