document.addEventListener("DOMContentLoaded", () => {
  const submitButton = document.getElementById("submit");
  const inputField = document.getElementById("input");

  submitButton.addEventListener("click", (e) => {
    let input = inputField.value;
    if (input === null || input === undefined || input.trim() === ''){
      return
    }

    inputField.value = "";

    output(input);

    autoResize(inputField)
  });

});

function output(input) {
  let product;

  // Regex remove non word/space chars
  // Trim trailing whitespce
  // Remove digits - not sure if this is best
  // But solves problem of entering something like 'hi1'

  let text = input.toLowerCase().replace(/[^\w\s]/gi, "").replace(/[\d]/gi, "").trim();
  text = text
    .replace(/ a /g, " ")   // 'tell me a story' -> 'tell me story'
    .replace(/i feel /g, "")
    .replace(/whats/g, "what is")
    .replace(/please /g, "")
    .replace(/ please/g, "")
    .replace(/r u/g, "are you");

  if (compare(prompts, replies, text)) { 
    // Search for exact match in `prompts`
    product = compare(prompts, replies, text);
  } else if (text.match(/thank/gi)) {
    product = "You're welcome!"
  } else if (text.match(/(corona|covid|virus)/gi)) {
    // If no match, check if message contains `coronavirus`
    product = coronavirus[Math.floor(Math.random() * coronavirus.length)];
  } else {
    // If all else fails: random alternative
    product = alternative[Math.floor(Math.random() * alternative.length)];
  }

  // Update DOM
  addChat(input, product);
}

function compare(promptsArray, repliesArray, string) {
  let reply;
  let replyFound = false;
  for (let x = 0; x < promptsArray.length; x++) {
    for (let y = 0; y < promptsArray[x].length; y++) {
      if (promptsArray[x][y] === string) {
        let replies = repliesArray[x];
        reply = replies[Math.floor(Math.random() * replies.length)];
        replyFound = true;
        // Stop inner loop when input value matches prompts
        break;
      }
    }
    if (replyFound) {
      // Stop outer loop when reply is found instead of interating through the entire array
      break;
    }
  }
  return reply;
}

function addChat(input, product) {
  const messagesContainer = document.getElementById("messages");

  // user message
  let userDiv = document.createElement("div");
  userDiv.id = "user";
  userDiv.className = "user message-container";
  userDiv.innerHTML = `<div class="user-message-body">${input}</div>
  <div><img src="user.png" class="avatar"></div>`;
  messagesContainer.appendChild(userDiv);

  // bot message
  let botDiv = document.createElement("div");
  botDiv.id = "bot";
  botDiv.className = "bot message-container";
  botDiv.innerHTML = `<div><img src="bot-mini.png" class="avatar"></div>
  <div class="bot-message-body">Thinking...</div>`;
  messagesContainer.appendChild(botDiv);

  // Keep messages at most recent
  messagesContainer.scrollTop = messagesContainer.scrollHeight - messagesContainer.clientHeight;

  setTimeout(() => {
    const messageBody = botDiv.getElementsByClassName("bot-message-body");
    console.log(messageBody);
    // messageBody[0].innerText = `${product}`;
    messageBody[0].innerText = `An response from bot`;
  }, 2000);


}

function autoResize(textarea) {
  textarea.style.height = 'auto'; // reset height
  textarea.style.height = textarea.scrollHeight + 'px'; // reset height according to content
}