<template>
  <footer class="footer">
    <p>© All rights reserved</p>
  </footer>
</template>

<script>
export default {
  name: 'Footer'
}
</script>

<style scoped>
.footer {
  background-color: black;
  color: white;
  text-align: center;
  padding: 10px 0;
  position: fixed;
  width: 100%;
  bottom: 0;
}
</style>

























<template>
  <div class="container">
    <header>
      <h1>Классификация родовых знаков</h1>
    </header>
    <div class="image-section">
      <img :src="mainImage" alt="Main" class="main-image" />
      <div class="additional-images">
        <img v-for="image in additionalImages" :src="image" :key="image" alt="Additional" class="additional-image" />
      </div>
      <div class="classification">
        <button @click="goBack">Вернуться</button>
        <p>Предсказанный класс: 1</p>
      </div>
    </div>
    <footer>
      <p>© All rights reserved</p>
    </footer>
  </div>
</template>

<script>
export default {
  data() {
    return {
      mainImage: require('@/assets/main-image.png'), // adjust the path as necessary
      additionalImages: [
        require('@/assets/image1.png'),
        require('@/assets/image2.png'),
        require('@/assets/image3.png')
      ]
    };
  },
  methods: {
    goBack() {
      this.$router.push('/'); // Adjust the route as necessary to return to the first page
    }
  }
};
</script>

<style scoped>
.container {
  text-align: center;
  font-family: Arial, sans-serif;
}

header {
  background-color: black;
  color: white;
  padding: 10px 0;
}

.image-section {
  margin: 20px;
}

.main-image {
  width: 600px;
  height: auto;
}

.additional-images {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-top: 10px;
}

.additional-image {
  width: 150px;
  height: auto;
}

.classification {
  margin-top: 20px;
}

button {
  background-color: black;
  color: white;
  padding: 10px 20px;
  border: none;
  cursor: pointer;
}

footer {
  margin-top: 20px;
}
</style>
