export async function compareModels(imageFile) {
  const formData = new FormData();
  formData.append("image", imageFile);

  const res = await fetch("http://127.0.0.1:5000/compare", {
    method: "POST",
    body: formData
  });

  return res.json();
}
