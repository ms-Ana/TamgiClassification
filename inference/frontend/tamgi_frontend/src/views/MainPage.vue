<script>
import Header from './components/Header.vue';
import Footer from './components/Footer.vue';
import MainContent from './components/MainContent.vue';

export default {
  name: 'MainPage',
  components: {
    Header,
    Footer,
    MainContent
  }
}
</script>

<style>
body, html, #app {
  margin: 0;
  padding: 0;
  height: 100%;
  font-family: Arial, sans-serif;
}
</style>
