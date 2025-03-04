document.getElementById("aqi-form").addEventListener("submit", function (e) {
  e.preventDefault();

  // Collect form data
  let formData = {};
  let valid = true;

  // Loop through the form inputs to collect and validate data
  document.querySelectorAll("#aqi-form input").forEach((input) => {
    const value = parseFloat(input.value);

    // Check if input is not empty and is a valid number
    if (isNaN(value) || value === "") {
      alert(`Invalid input for ${input.name}`);
      valid = false;
      return;
    }

    formData[input.name] = value;
  });

  if (!valid) {
    return; // Prevent form submission if invalid input
  }

  // Send data to the backend
  fetch("http://localhost:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(formData),
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("result").textContent =
        "Predicted AQI: " + data.predicted_AQI.toFixed(2);
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("Error occurred while predicting AQI. Please try again.");
    });
});
