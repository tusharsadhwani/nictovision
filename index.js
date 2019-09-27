console.log("Hello")

function read_file(file) {
  var input = file.target

  var reader = new FileReader()
  reader.onload = function(){
    var dataURL = reader.result
    console.log(dataURL)
  }
  img_data = reader.readAsDataURL(input.files[0])
  console.log(img_data)
}