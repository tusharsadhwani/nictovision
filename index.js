
fetch('https://api.github.com/users/oswalgopal').then(res => res.json()).then(res => {
    console.log(res);
}).catch(err => {
    console.error(err);
})