// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyBf78aNLUckBIq9gKDJV2B1wDIdZdXp4bg",
  authDomain: "my-safety-app-55906.firebaseapp.com",
  projectId: "my-safety-app-55906",
  // storageBucket: "my-safety-app-55906.firebasestorage.app",
  storageBucket: "my-safety-app-55906.appspot.com",
  messagingSenderId: "685832140461",
  appId: "1:685832140461:web:a687cdc409c82f02513824",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);
export default app;