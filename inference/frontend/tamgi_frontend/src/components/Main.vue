<template>
  <div class="main-content">
    <button @click="uploadImage">Загрузить изображение</button>
    <div class="drop-area" @dragover.prevent @drop.prevent="dropImage">
      <p>ИЛИ ПЕРЕТАЩИТЕ ИЗОБРАЖЕНИЕ СЮДА</p>
    </div>
    <button @click="classifyImage">Классифицировать</button>
  </div>
</template>

<script>
шmport { defineComponent } from 'vue';
import axios from 'axios';

export default defineComponent({
  name: 'ImageUploader',
  data() {
    return {
      imageFile: null as File | null,
      temporaryDirectory: 'path/to/temp/dir'
    };
  },
  methods: {
    uploadImage(): void {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/*';
      input.onchange = () => {
        if (input.files && input.files[0]) {
          this.imageFile = input.files[0];
          this.saveImageLocally(this.imageFile);
        }
      };
      input.click();
    },
    dropImage(event: DragEvent): void {
      if (event.dataTransfer && event.dataTransfer.files[0]) {
        this.imageFile = event.dataTransfer.files[0];
        this.saveImageLocally(this.imageFile);
      }
    },
    saveImageLocally(file: File): void {
      const reader = new FileReader();
      reader.onload = (e: ProgressEvent<FileReader>) => {
        if (e.target && e.target.result) {
          const arrayBuffer = e.target.result as ArrayBuffer;
          const blob = new Blob([arrayBuffer]);
          const link = document.createElement('a');
          link.href = window.URL.createObjectURL(blob);
          link.download = `${this.temporaryDirectory}/${file.name}`;
          link.click();
        }
      };
      reader.readAsArrayBuffer(file);
    },
    async classifyImage(): Promise<void> {
      if (!this.imageFile) {
        alert('Please upload or drop an image first.');
        return;
      }
      try {
        const formData = new FormData();
        formData.append('imagePath', `${this.temporaryDirectory}/${this.imageFile.name}`);

        const response = await axios.post('http://localhost:8080/classify', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        });

        const data = response.data;
        console.log('Class:', data.class);
        console.log('Related images:', data.image_files);
      } catch (error) {
        console.error('Error classifying image:', error);
      }
    }
  }
});
</script>
</script>

<style scoped>
.main-content {
  text-align: center;
  margin: 50px auto;
}

button {
  background-color: black;
  color: white;
  border: none;
  padding: 10px 20px;
  cursor: pointer;
  margin: 10px;
}

.drop-area {
  border: 2px dashed black;
  padding: 50px;
  margin: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.drop-area p {
  font-size: 18px;
}
</style>
