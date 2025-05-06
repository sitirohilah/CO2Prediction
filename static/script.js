function predictCO2() {
    let year = document.getElementById("yearInput").value;
    if (!year) {
        alert("Masukkan tahun terlebih dahulu!");
        return;
    }

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ year: parseInt(year) })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = data.predicted_CO2.toFixed(2) + " metric tons";
        updateChart(year, data.predicted_CO2);
    })
    .catch(error => console.error("Error:", error));
}

function updateChart(year, predictedCO2) {
    let ctx = document.getElementById("co2Chart").getContext("2d");
    let chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [year],
            datasets: [{
                label: "Prediksi Emisi CO2",
                data: [predictedCO2],
                borderColor: "red",
                fill: false
            }]
        }
    });
}
