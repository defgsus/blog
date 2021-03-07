
console.log("CONTENT SCRIPT LOADED");


// TODO: This is currently not working,
//      does not receive messages from page
window.addEventListener("message", function(event) {
    console.log("CONTENT GET MESSAGE", event);
    if (event.source === window) {
        switch (event.data.type) {
            case "har-research-eval":
               eval_code(event.data.code)
                   .then(result => {
                       window.postMessage({
                           type: "har-research-eval-success",
                           result: result,
                       }, "*");
                   });
               break;
        }
    }
});


function eval_code(code) {
    return new Promise((resolve, reject) => {
        try {
            resolve(eval(code));
        }
        catch (e) {
            reject(e);
        }
    });
}