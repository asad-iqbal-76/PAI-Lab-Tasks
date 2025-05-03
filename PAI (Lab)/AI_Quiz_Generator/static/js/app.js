document.getElementById('questionForm').addEventListener('submit', async function (e) {
    e.preventDefault();
    const topicInput = document.getElementById('topic').value.trim();
    const fileInput = document.getElementById('file');
    const difficulty = document.getElementById('difficulty').value;
    const qtype = document.getElementById('qtype').value;
    const num_questions = document.getElementById('num_questions').value;
    const container = document.getElementById('questionsContainer');
    container.innerHTML = '';
    document.getElementById('spinner').classList.remove('hidden');
    document.getElementById('generateBtn').disabled = true;
    try {
        if (!topicInput && fileInput.files.length === 0) {
            throw new Error("Please enter a topic or upload a file.");
        }
        let context = topicInput;
        if (fileInput.files.length > 0) {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
  
            const uploadResponse = await fetch('/upload', { method: 'POST', body: formData });
            const uploadData = await uploadResponse.json();
  
            if (uploadData.error) throw new Error(uploadData.error);
            context = uploadData.context;
        }
        const response = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ topic: topicInput, context, difficulty, qtype, num_questions })
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error);
        data.questions.forEach((q, i) => {
            const card = document.createElement('div');
            card.className = 'card animate-slide-up';
            card.style.animationDelay = `${i * 0.1}s`;
            const cardBody = document.createElement('div');
            cardBody.className = 'card-body';
            const title = document.createElement('h5');
            title.className = 'card-title';
            title.textContent = `${i + 1}. ${q.question}`;
            cardBody.appendChild(title);
            if (q.type === 'MCQ') {
                const ul = document.createElement('ul');
                ul.className = 'list-group list-group-flush';
                q.options.forEach(opt => {
                    const li = document.createElement('li');
                    li.className = 'list-group-item';
                    li.innerHTML = opt === q.answer ? `<strong>${opt}</strong>` : opt;
                    ul.appendChild(li);
                });
                cardBody.appendChild(ul);
            } else {
                const para = document.createElement('p');
                para.innerHTML = `<strong>Answer:</strong> ${q.answer || 'Not provided'}`;
                cardBody.appendChild(para);
            }
            card.appendChild(cardBody);
            container.appendChild(card);
        });
        const btnGroup = document.createElement('div');
        btnGroup.className = 'mt-8 flex space-x-4 animate-slide-up';
        const jsonBtn = document.createElement('button');
        jsonBtn.className = 'btn-success';
        jsonBtn.textContent = 'Export as JSON';
        jsonBtn.onclick = () => {
            const blob = new Blob([JSON.stringify(data.questions, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'quiz.json';
            a.click();
            URL.revokeObjectURL(url);
        };
        const pdfBtn = document.createElement('button');
        pdfBtn.className = 'btn-danger';
        pdfBtn.textContent = 'Download as PDF';
        pdfBtn.onclick = () => {
            const formData = new FormData();
            formData.append('topic', topicInput);
            formData.append('questions', JSON.stringify(data.questions));
            fetch('/download_pdf', {
                method: 'POST',
                body: formData
            })
                .then(res => {
                    if (!res.ok) throw new Error("Failed to generate PDF.");
                    return res.blob();
                })
                .then(blob => {
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'quiz.pdf';
                    a.click();
                    URL.revokeObjectURL(url);
                })
                .catch(err => {
                    const alert = document.createElement('div');
                    alert.className = 'alert animate-slide-up';
                    alert.textContent = `Error downloading PDF: ${err.message}`;
                    container.appendChild(alert);
                });
        };
        btnGroup.appendChild(jsonBtn);
        btnGroup.appendChild(pdfBtn);
        container.appendChild(btnGroup);
    } catch (err) {
        const alert = document.createElement('div');
        alert.className = 'alert animate-slide-up';
        alert.textContent = `Error: ${err.message}`;
        container.appendChild(alert);
    } finally {
        document.getElementById('spinner').classList.add('hidden');
        document.getElementById('generateBtn').disabled = false;
    }
  });