from flask import render_template, request, url_for, flash, redirect, session
from flaskblog import app
from flaskblog.forms import RegistrationForm, LoginForm
from gensim.models.fasttext import FastText
import pandas as pd
import pickle
import os
from bs4 import BeautifulSoup

posts = [
    {
        'Title': 'Assembly/Production Technicians Milton Keynes',
        'Company': 'Newstaff Employment Services Ltd',
        'Description': 'Main Purpose of Job:To perform a range of mechanical assembly, hydraulic installation and electrical wiring manufacturing / production operations to the required quality standard, within cost targets. Main Tasks Of Job: Efficient and correct to specification Mechanical assembly, hydraulic installation, electrical wiring, and fabrication / fitting operations, linked to the manufacture, service and repair of access platforms. Competent build inspection of all stages and materials used. Full compliance with ISO 9001 procedures, and total quality focus throughout all direct / indirect stages of manufacturing process to achieve a quality product aligned to the customers specification / expectations. Effective and efficient value added organisation and control of all allocated operations. Full flexibility, including being training and training others to enable all manufacturing operations by all to be completed across the company. Compliance with health and safety standards, flexibility, and development of skills through the complete manufacturing process, and the production / product cells. Maintaining and developing accurate systems and procedures. Ensuring engineering masters and works orders accurately reflect the approved parts consumed. Identification of low stock / shortages. Maintain a high standard of **** s workplace organisation. Continuous improvement / elimination of waste, through Lean Manufacturing techniques / tools . Working as a team to achieve customer quality, delivery, cost and innovation requirements. Striving to meet performance requirements for accidents / incidents, internal quality defects,cell productivity / utilisation, overall stock accuracy and WIP (flow lineadherence). Ensuring company procedures are complied with. Qualifications And Experience: Ideally City and Guilds formal mechanical or electrical assembly qualification,although stable and proven experience in a previous mechanical / electrical role will be considered where formal qualifications do not exist. Proven and communicated hands on experience and skills in the manufacturing practices of medium to heavy engineering assembly, hydraulic installation and basic wiring. Able to achieve strict quality standards and specifications. Team player, with full flexibility and availability for regular overtime working (where business demand requires) and performance of skilled and un skilled roles / tasks. Ability to work under pressure and to meet deadlines. Remuneration: The above role offers a competitive wage rate, business performance bonus, healthcare scheme, contributory pension, life insurance and holiday allowance scheme',
    },
    {
        'Title': 'Assembly/Production Technicians Milton Keynes',
        'Company': 'Newstaff Employment Services Ltd',
        'Description': 'Main Purpose of Job:To perform a range of mechanical assembly, hydraulic installation and electrical wiring manufacturing / production operations to the required quality standard, within cost targets. Main Tasks Of Job: Efficient and correct to specification Mechanical assembly, hydraulic installation, electrical wiring, and fabrication / fitting operations, linked to the manufacture, service and repair of access platforms. Competent build inspection of all stages and materials used. Full compliance with ISO 9001 procedures, and total quality focus throughout all direct / indirect stages of manufacturing process to achieve a quality product aligned to the customers specification / expectations. Effective and efficient value added organisation and control of all allocated operations. Full flexibility, including being training and training others to enable all manufacturing operations by all to be completed across the company. Compliance with health and safety standards, flexibility, and development of skills through the complete manufacturing process, and the production / product cells. Maintaining and developing accurate systems and procedures. Ensuring engineering masters and works orders accurately reflect the approved parts consumed. Identification of low stock / shortages. Maintain a high standard of **** s workplace organisation. Continuous improvement / elimination of waste, through Lean Manufacturing techniques / tools . Working as a team to achieve customer quality, delivery, cost and innovation requirements. Striving to meet performance requirements for accidents / incidents, internal quality defects,cell productivity / utilisation, overall stock accuracy and WIP (flow lineadherence). Ensuring company procedures are complied with. Qualifications And Experience: Ideally City and Guilds formal mechanical or electrical assembly qualification,although stable and proven experience in a previous mechanical / electrical role will be considered where formal qualifications do not exist. Proven and communicated hands on experience and skills in the manufacturing practices of medium to heavy engineering assembly, hydraulic installation and basic wiring. Able to achieve strict quality standards and specifications. Team player, with full flexibility and availability for regular overtime working (where business demand requires) and performance of skilled and un skilled roles / tasks. Ability to work under pressure and to meet deadlines. Remuneration: The above role offers a competitive wage rate, business performance bonus, healthcare scheme, contributory pension, life insurance and holiday allowance scheme',
    }
]


def gen_docVecs(wv, tk_txts):  # generate vector representation for documents
    docs_vectors = pd.DataFrame()  # creating empty final dataframe
    # stopwords = nltk.corpus.stopwords.words('english') # if we haven't pre-processed the articles, it's a good idea to remove stop words

    for i in range(0, len(tk_txts)):
        tokens = tk_txts[i]
        temp = pd.DataFrame()  # creating a temporary dataframe(store value for 1st doc & for 2nd doc remove the details of 1st & proced through 2nd and so on..)
        for w_ind in range(0, len(tokens)):  # looping through each word of a single document and spliting through space
            try:
                word = tokens[w_ind]
                word_vec = wv[
                    word]  # if word is present in embeddings(goole provides weights associate with words(300)) then proceed
                temp = temp.append(pd.Series(word_vec),
                                   ignore_index=True)  # if word is present then append it to temporary dataframe
            except:
                pass
        doc_vector = temp.sum()  # take the sum of each column
        docs_vectors = docs_vectors.append(doc_vector,
                                           ignore_index=True)  # append each document value to the final dataframe
    return docs_vectors


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', posts=posts)


@app.route("/about")
def about():
    return render_template('about.html', title='About')


@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('home'))
    return render_template('register.html', title='Register', form=form)


@app.route('/Accounting_Finance')
def Accounting_Finance():
    return render_template('Accounting_Finance.html')


@app.route('/Engineering')
def Engineering():
    return render_template('Engineering.html')


@app.route('/Healthcare_Nursing')
def Healthcare_Nursing():
    return render_template('Healthcare_Nursing.html')


@app.route('/Sales')
def Sales():
    return render_template('Sales.html')


@app.route('/<folder>/<filename>')
def job_ad(folder, filename):
    return render_template('/' + folder + '/' + filename + '.html')


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if 'username' in session:
        if request.method == 'POST':

            # Read the content
            f_title = request.form['title']
            f_content = request.form['description']

            # Classify the content
            if request.form['button'] == 'Classify':

                # Tokenize the content of the .txt file so as to input to the saved model
                # Here, as an example,  we just do a very simple tokenization
                tokenized_data = f_content.split(' ')

                # Load the FastText model
                bbcFT = FastText.load("bbcFT.model")
                bbcFT_wv = bbcFT.wv

                # Generate vector representation of the tokenized data
                bbcFT_dvs = gen_docVecs(bbcFT_wv, [tokenized_data])

                # Load the LR model
                pkl_filename = "bbcFT_LR.pkl"
                with open(pkl_filename, 'rb') as file:
                    model = pickle.load(file)

                # Predict the label of tokenized_data
                y_pred = model.predict(bbcFT_dvs)
                y_pred = y_pred[0]

                return render_template('admin.html', prediction=y_pred, title=f_title, description=f_content)

            elif request.form['button'] == 'Save':
                # First check if the recommended category is empty
                cat_recommend = request.form['category']
                if cat_recommend == '':
                    return render_template('admin.html', prediction=cat_recommend,
                                           title=f_title, description=f_content,
                                           category_flag='Recommended category must not be empty.')

                elif cat_recommend not in ['Engineering', 'Accounting_Finance', 'Healthcare_Nursing', 'Sales']:
                    return render_template('admin.html', prediction=cat_recommend,
                                           title=f_title, description=f_content,
                                           category_flag='Recommended category must belong to: Engineering, Accounting_Finance, Healthcare_Nursing, Sales.')

                else:

                    # First read the html template
                    soup = BeautifulSoup(open('templates/job_ad_template.html'), 'html.parser')

                    # Then adding the title and the content to the template
                    # First, add the title
                    div_page_title = soup.find('div', {'class': 'title'})
                    title = soup.new_tag('h1', id='data-title')
                    title.append(f_title)
                    div_page_title.append(title)

                    # Second, add the content
                    div_page_content = soup.find('div', {'class': 'data-article'})
                    content = soup.new_tag('p')
                    content.append(f_content)
                    div_page_content.append(content)

                    # Finally write to a new html file
                    filename_list = f_title.split()
                    filename = '_'.join(filename_list)
                    filename = cat_recommend + '/' + filename + ".html"
                    with open("templates/" + filename, "w", encoding='utf-8') as file:
                        print(filename)
                        file.write(str(soup))

                    # Redirect to the newly-generated news article
                    return redirect('/' + filename.replace('.html', ''))

        else:
            return render_template('admin.html')

    else:
        return redirect('/login')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if 'username' in session:
        return redirect(url_for('admin'))
    else:
        if form.validate_on_submit():
            if form.email.data == 'ngocanhthu20102002@gmail.com' and form.password.data == 'tnathu-ai':
                flash('You have been logged in!', 'Success')
                return redirect(url_for('admin'))
            else:
                flash('Login Unsuccessful. Invalid username or password', 'danger')
        return render_template('login.html', title='Login', form=form)


@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)

    return redirect('/')

