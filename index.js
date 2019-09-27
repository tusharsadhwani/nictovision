function read_file(file) {
  var input = file.target

  var reader = new FileReader()
  reader.onload = function(){
    let dataURL = reader.result
    let uploaded_img = document.getElementById('uploaded_img')
    uploaded_img.src = dataURL
  }
  img_data = reader.readAsDataURL(input.files[0])
} 