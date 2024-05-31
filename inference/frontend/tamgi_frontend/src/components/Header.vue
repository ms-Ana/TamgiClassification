<template>
  <header class="header">
    <h1>Классификация родовых знаков</h1>
  </header>
</template>

<script>
export default {
  name: 'Header'
}
</script>

<style scoped>
.header {
  background-color: black;
  color: white;
  text-align: center;
  padding: 20px 0;
}
</style>






























<template>
  <div class="container">
    <header>
      <h1>Классификация родовых знаков</h1>
    </header>
    <div v-if="!classified" class="upload-section">
      <button @click="triggerFileUpload">Загрузить изображение</button>
      <input type="file" ref="fileInput" @change="handleFileUpload" style="display: none" />
      <div
        class="drop-area"
        @dragover.prevent
        @drop.prevent="handleDrop"
      >
        ИЛИ ПЕРЕТАЩИТЕ ИЗОБРАЖЕНИЕ СЮДА
      </div>
      <button @click="classifyImage">Классифицировать</button>
    </div>
    <div v-else class="image-section">
      <img :src="mainImage" alt="Main" class="main-image" />
      <div class="additional-images">
        <img v-for="image in additionalImages" :src="image" :key="image" alt="Additional" class="additional-image" />
      </div>
      <div class="classification">
        <button @click="goBack">Вернуться</button>
        <p>Предсказанный класс: {{ predictedClass }}</p>
      </div>
    </div>
    <footer>
      <p>© All rights reserved</p>
    </footer>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      mainImage: '',
      additionalImages: [],
      predictedClass: null,
      classified: false
    };
  },
  methods: {
    triggerFileUpload() {
      this.$refs.fileInput.click();
    },
    handleFileUpload(event) {
      const file = event.target.files[0];
      this.uploadFile(file);
    },
    handleDrop(event) {
      const file = event.dataTransfer.files[0];
      this.uploadFile(file);
    },
    uploadFile(file) {
      // Handle the file upload logic here
      const reader = new FileReader();
      reader.onload = (e) => {
        this.mainImage = e.target.result;
      };
      reader.readAsDataURL(file);
    },
    async classifyImage() {
      try {
        const formData = new FormData();
        formData.append('image', this.$refs.fileInput.files[0]);

        const response = await axios.post('YOUR_BACKEND_API_URL', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });

        const data = response.data;
        this.predictedClass = data.class;
        this.additionalImages = data.images;
        this.classified = true;
      } catch (error) {
        console.error('Error classifying image:', error);
      }
    },
    goBack() {
      this.classified = false;
      this.mainImage = '';
      this.additionalImages = [];
      this.predictedClass = null;
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

.upload-section {
  margin: 20px;
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
