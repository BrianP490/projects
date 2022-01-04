const canvas = document.getElementById('game'); //refer the canvas to draw on
const ctx = canvas.getContext('2d'); //2d context to draw onto the canvas

//class for the snake parts consisting of x and y variables
class SnakePart {
    constructor(x, y) {
        this.x = x;     //x coordinate of snake part
        this.y = y;     //x coordinate of snake part
    }
}

let speed = 7;      //speed of the snake

let tileCount = 20;     //number of tiles per side
let tileSize = canvas.width / tileCount - 2;        //size of a tile
let headX = 10; //starting x-position of the snake
let headY = 10; //starting x-position of the snake
const snakeParts = [] //array of SnakePart classes
let tailLength = 2; //snake has this many tail parts

let appleX = 5; //starting x-position of the APPLE
let appleY = 5; //starting Y-position of the APPLE

let inputsXVelocity = 0;    //starting snake  input X-velocity
let inputsYVelocity = 0;    //starting snake input y-velocity

let xVelocity = 0;  //starting snake X-velocity
let yVelocity = 0;  //starting snake y-velocity

let score = 0;  //user score;initially set to 0

const gulpSound = new Audio("gulp.mp3");    //assign the audio to a variable

//game loop
function drawGame() {
    xVelocity = inputsXVelocity;
    yVelocity = inputsYVelocity;

    changeSnakePosition();  //function call to move the snake
    let result = isGameOver();  //check gameover function; returns boolean
    if(result){
        return; //end the game
    }

    clearScreen();  //refresh the screen

    checkAppleCollision();  //check if the apple was eaten by the snake
    drawApple();    //
    drawSnake();

    drawScore();

    setTimeout(drawGame, 1000/speed);   //wait to recall drawGame() and update the game in the next frame in ___ ms delay
}

function isGameOver() {
    let gameOver = false; //variable to check game state

    if(yVelocity === 0 && xVelocity === 0)//return to loop and skip rest of checks if the game is just starting out
        return false;

    //checks for wall collisions
    if(headX < 0 || headX === tileCount || headY < 0 || headY === tileCount) {
        gameOver = true;
    }

    //checks for snake collision
    for (let i = 0; i < snakeParts.length; i++) {//loop through the snake array
        let part = snakeParts[i];
        if (part.x === headX && part.y === headY) { //if snake collided with itself
            gameOver = true;
            break; //break out of the for loop
        }
    }

    if(gameOver) { //if we have a game end state
        ctx.fillStyle = 'white';    //next two lines style the text
        ctx.font = '50px Verdana';
        //add a retro themed "game over" message
        var gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
        gradient.addColorStop("0", " magenta");
        gradient.addColorStop("0.5", "blue");
        gradient.addColorStop("1.0", "red");
        // Fill with gradient
        ctx.fillStyle = gradient;

        ctx.fillText('Game Over!', canvas.width / 6.5 , canvas.height / 2 );
    }
    return gameOver;    //return
}
//print the score to the screen
function drawScore() {
    ctx.fillStyle = 'white';    //next 2 lines style the text
    ctx.font = '10px Verdana';
    ctx.fillText("Score: " + score, canvas.width - 50, 10);
}
//clear the screen
function clearScreen() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}
//draw the snake
function drawSnake() {
    ctx.fillStyle = 'green'
    for(let i = 0; i < snakeParts.length; i++) {
        ctx.fillRect(snakeParts[i].x * tileCount, snakeParts[i].y * tileCount, tileSize, tileSize) //fill in a rectangle that resenbles the snake's body
    }

    snakeParts.push(new SnakePart(headX, headY)); //put an item at the end of the list next to the head; saves a new head block

    while(snakeParts.length > tailLength) { //while the head is still there
        snakeParts.shift(); //remove the first item from the snake parts if it has more than our tail size; removes head
    }
    //draw the head
    ctx.fillStyle = 'orange';
    ctx.fillRect(headX * tileCount, headY * tileCount, tileSize, tileSize) //draws the snake head
}

//update snake position
function changeSnakePosition() {
    headX = headX + xVelocity;
    headY = headY + yVelocity;
}
//draw the apple
function drawApple() {
    ctx.fillStyle = 'red';
    ctx.fillRect(appleX * tileCount, appleY * tileCount, tileSize, tileSize);//draw the apple at its position
}
//check snake to apple collision
function checkAppleCollision() {
    if(appleX === headX && appleY === headY) {//if we have a collision; find a new spot for the apple
        appleX = Math.floor(Math.random() * tileCount); //generates a number between 0 and 1, multiplies by tilecount and rounds down for new x position
        appleY = Math.floor(Math.random() * tileCount);
        tailLength++; //increase the tail length for eating the apple
        score++;    //increase the player's score
        gulpSound.play();   //play an audible audio file
    }
}

//react to user input
document.body.addEventListener('keydown', keyDown);//whenever the player presses a key, call the KeyDown function to update the snake's direction

function keyDown(event) {
    //w- up key
    if(event.keyCode == 87) {
        if(inputsYVelocity == 1)
            return;
        inputsYVelocity = -1;
        inputsXVelocity = 0;
    }
    //s- down key
    if(event.keyCode == 83) {
        if(inputsYVelocity == -1)
            return;
        inputsYVelocity = 1;
        inputsXVelocity = 0;
    }
    //a- left left
    if(event.keyCode == 65) {
        if(inputsXVelocity == 1)
            return;
        inputsYVelocity = 0;
        inputsXVelocity = -1;
    }
    //d- right key
    if(event.keyCode == 68) {
        if(inputsXVelocity == -1)
            return;
        inputsYVelocity = 0;
        inputsXVelocity = 1;
    }
}

drawGame();  //start the game