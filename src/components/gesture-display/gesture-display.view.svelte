<script>
  import { ViewContainer } from '@marcellejs/design-system';
  export let link;
  export let img;
  export let title;
  export let gesture;
  export let completed;
</script>

<ViewContainer {title}>
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
  {#if $link !== undefined}
    <img id="gesture" src={$link} alt="" style="height: 100px; margin: 5px auto" />
  {/if}
</ViewContainer>

<style>
  .centered-element {
    text-align: center;
    display: inline-block;
  }

  .gesture {
    font-size: 30px;
    color: black;
    text-align: center;
  }
  .completed {
    font-size: 30px;
    color: rgb(255, 2, 2);
    text-align: center;
  }
</style>
