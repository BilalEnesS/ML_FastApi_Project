// Tab switching functionality
document.addEventListener('DOMContentLoaded', function() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            // Remove active class from all tabs and contents
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked tab and corresponding content
            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        });
    });
});

// Helper function to show results
function showResult(elementId, message, type = 'info') {
    const element = document.getElementById(elementId);
    element.innerHTML = `<pre>${JSON.stringify(message, null, 2)}</pre>`;
    element.className = `result-area ${type}`;
}

// File upload functionality
async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showResult('uploadResult', { error: 'Lütfen bir dosya seçin' }, 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showResult('uploadResult', result, 'success');
        } else {
            showResult('uploadResult', result, 'error');
        }
    } catch (error) {
        showResult('uploadResult', { error: error.message }, 'error');
    }
}

// Save configuration
async function saveConfig() {
    const algorithm = document.getElementById('algorithm').value;
    const hyperparamsText = document.getElementById('hyperparameters').value;
    
    let hyperparameters = {};
    if (hyperparamsText.trim()) {
        try {
            hyperparameters = JSON.parse(hyperparamsText);
        } catch (e) {
            showResult('configResult', { error: 'Geçersiz JSON formatı' }, 'error');
            return;
        }
    }

    const config = { algorithm, hyperparameters };

    try {
        const response = await fetch('/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showResult('configResult', result, 'success');
        } else {
            showResult('configResult', result, 'error');
        }
    } catch (error) {
        showResult('configResult', { error: error.message }, 'error');
    }
}

// Start training
async function startTraining() {
    try {
        const response = await fetch('/train', {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showResult('trainResult', result, 'success');
        } else {
            showResult('trainResult', result, 'error');
        }
    } catch (error) {
        showResult('trainResult', { error: error.message }, 'error');
    }
}

// Make prediction
async function makePrediction() {
    const data = {
        company_code: document.getElementById('companyCode').value,
        document_number: document.getElementById('documentNumber').value,
        description: document.getElementById('description').value,
        payment_type: document.getElementById('paymentType').value,
        amount: parseFloat(document.getElementById('amount').value),
        currency_code: document.getElementById('currencyCode').value,
        transaction_type: document.getElementById('transactionType').value
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showResult('predictResult', result, 'success');
        } else {
            showResult('predictResult', result, 'error');
        }
    } catch (error) {
        showResult('predictResult', { error: error.message }, 'error');
    }
}

// Load metrics
async function loadMetrics() {
    try {
        const response = await fetch('/metrics');
        const result = await response.json();
        
        if (response.ok) {
            showResult('metricsResult', result, 'info');
        } else {
            showResult('metricsResult', result, 'error');
        }
    } catch (error) {
        showResult('metricsResult', { error: error.message }, 'error');
    }
}
