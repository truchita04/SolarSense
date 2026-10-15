// ============================================================
// SOLARSENSE FRONTEND
// ============================================================

const API_URL = "http://127.0.0.1:8000";

let predictionChart = null;
let featureImportanceChart = null;


// ============================================================
// GET INPUT DATA
// ============================================================

function getInputData() {

    const irradiance =
        parseFloat(
            document.getElementById("irradiance").value
        );

    const ambientTemperature =
        parseFloat(
            document.getElementById("ambient_temperature").value
        );

    const moduleTemperature =
        parseFloat(
            document.getElementById("module_temperature").value
        );

    const hour =
        parseInt(
            document.getElementById("hour").value
        );

    const minute =
        parseInt(
            document.getElementById("minute").value
        );

    const day =
        parseInt(
            document.getElementById("day").value
        );

    const month =
        parseInt(
            document.getElementById("month").value
        );

    const dayOfWeek =
        parseInt(
            document.getElementById("day_of_week").value
        );

    const lagPower =
        parseFloat(
            document.getElementById("lag_power").value
        );

    const rollingPower =
        parseFloat(
            document.getElementById("rolling_power").value
        );


    return {

        ambient_temperature:
            ambientTemperature,

        module_temperature:
            moduleTemperature,

        irradiance:
            irradiance,

        hour:
            hour,

        minute:
            minute,

        day:
            day,

        month:
            month,

        day_of_week:
            dayOfWeek,

        is_day:
            irradiance > 0 ? 1 : 0,

        lag_power:
            lagPower,

        rolling_power:
            rollingPower
    };
}


// ============================================================
// PREDICT POWER
// ============================================================

async function predictPower() {

    const resultElement =
        document.getElementById(
            "predictionResult"
        );

    const modelElement =
        document.getElementById(
            "predictionModel"
        );


    resultElement.innerText =
        "Calculating...";


    try {

        const inputData =
            getInputData();


        const response =
            await fetch(
                `${API_URL}/api/predict`,
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            inputData
                        )
                }
            );


        if (!response.ok) {

            throw new Error(
                "Prediction request failed"
            );

        }


        const result =
            await response.json();


        resultElement.innerText =
            `${result.predicted_power.toFixed(2)} kW`;


        modelElement.innerText =
            `Model: ${result.model}`;


    } catch (error) {

        console.error(error);

        resultElement.innerText =
            "Prediction failed";

        modelElement.innerText =
            "Make sure FastAPI is running.";

    }
}


// ============================================================
// DETECT ANOMALY
// ============================================================

async function detectFault() {

    const resultElement =
        document.getElementById(
            "faultResult"
        );

    const scoreElement =
        document.getElementById(
            "anomalyScore"
        );


    resultElement.innerHTML = `
        <div class="status-icon">
            ⏳
        </div>

        <div>

            <strong>
                Analyzing...
            </strong>

            <p>
                Checking operating conditions.
            </p>

        </div>
    `;


    try {

        const inputData =
            getInputData();


        const faultPower =
            parseFloat(
                document.getElementById(
                    "fault_power"
                ).value
            );


        inputData.lag_power =
            faultPower;


        const response =
            await fetch(
                `${API_URL}/api/detect-fault`,
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(
                            inputData
                        )
                }
            );


        if (!response.ok) {

            throw new Error(
                "Anomaly detection failed"
            );

        }


        const result =
            await response.json();


        const isAnomaly =
            result.status.includes(
                "ANOMALY"
            );


        if (isAnomaly) {

            resultElement.className =
                "status-box anomaly";


            resultElement.innerHTML = `

                <div class="status-icon">
                    ⚠️
                </div>

                <div>

                    <strong>
                        POSSIBLE ANOMALY
                    </strong>

                    <p>
                        The operating condition may
                        require further inspection.
                    </p>

                </div>

            `;

        } else {

            resultElement.className =
                "status-box normal";


            resultElement.innerHTML = `

                <div class="status-icon">
                    🟢
                </div>

                <div>

                    <strong>
                        NORMAL
                    </strong>

                    <p>
                        No abnormal operating condition
                        detected.
                    </p>

                </div>

            `;

        }


        scoreElement.innerText =
            result.anomaly_score.toFixed(4);


    } catch (error) {

        console.error(error);


        resultElement.className =
            "status-box anomaly";


        resultElement.innerHTML = `

            <div class="status-icon">
                ❌
            </div>

            <div>

                <strong>
                    ERROR
                </strong>

                <p>
                    Unable to connect to the API.
                </p>

            </div>

        `;


        scoreElement.innerText =
            "--";

    }
}


// ============================================================
// ACTUAL VS PREDICTED CHART
// ============================================================

async function loadPredictionChart() {

    try {

        const response =
            await fetch(
                `${API_URL}/api/prediction-chart`
            );


        if (!response.ok) {

            throw new Error(
                "Chart data request failed"
            );

        }


        const data =
            await response.json();


        if (data.error) {

            console.error(
                data.error
            );

            return;

        }


        const canvas =
            document.getElementById(
                "predictionChart"
            );


        const context =
            canvas.getContext("2d");


        if (predictionChart) {

            predictionChart.destroy();

        }


        predictionChart =
            new Chart(
                context,
                {

                    type: "line",

                    data: {

                        labels:
                            data.dates,

                        datasets: [

                            {
                                label:
                                    "Actual Power",

                                data:
                                    data.actual,

                                borderWidth:
                                    2,

                                pointRadius:
                                    0,

                                tension:
                                    0.25
                            },

                            {
                                label:
                                    "Predicted Power",

                                data:
                                    data.predicted,

                                borderWidth:
                                    2,

                                pointRadius:
                                    0,

                                tension:
                                    0.25
                            }

                        ]

                    },

                    options: {

                        responsive:
                            true,

                        maintainAspectRatio:
                            false,

                        interaction: {

                            mode:
                                "index",

                            intersect:
                                false

                        },

                        scales: {

                            x: {

                                title: {

                                    display:
                                        true,

                                    text:
                                        "Date / Time"

                                }

                            },

                            y: {

                                title: {

                                    display:
                                        true,

                                    text:
                                        "Solar Power (kW)"

                                },

                                beginAtZero:
                                    true

                            }

                        }

                    }

                }
            );


    } catch (error) {

        console.error(
            "Prediction chart error:",
            error
        );

    }
}


// ============================================================
// FEATURE IMPORTANCE CHART
// ============================================================

async function loadFeatureImportance() {

    try {

        const response =
            await fetch(
                `${API_URL}/api/feature-importance`
            );


        if (!response.ok) {

            throw new Error(
                "Feature importance request failed"
            );

        }


        const data =
            await response.json();


        if (data.error) {

            console.error(
                data.error
            );

            return;

        }


        const canvas =
            document.getElementById(
                "featureImportanceChart"
            );


        const context =
            canvas.getContext("2d");


        if (featureImportanceChart) {

            featureImportanceChart.destroy();

        }


        featureImportanceChart =
            new Chart(
                context,
                {

                    type: "bar",

                    data: {

                        labels:
                            data.features,

                        datasets: [

                            {
                                label:
                                    "Feature Importance",

                                data:
                                    data.importance,

                                borderWidth:
                                    1
                            }

                        ]

                    },


                    options: {

                        indexAxis:
                            "y",

                        responsive:
                            true,

                        maintainAspectRatio:
                            false,

                        plugins: {

                            legend: {

                                display:
                                    false

                            }

                        },

                        scales: {

                            x: {

                                beginAtZero:
                                    true,

                                title: {

                                    display:
                                        true,

                                    text:
                                        "Importance"

                                }

                            },

                            y: {

                                title: {

                                    display:
                                        true,

                                    text:
                                        "Feature"

                                }

                            }

                        }

                    }

                }
            );


    } catch (error) {

        console.error(
            "Feature importance error:",
            error
        );

    }
}


// ============================================================
// PAGE LOAD
// ============================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        loadPredictionChart();

        loadFeatureImportance();

    }
);