# Project Tree of RAG for company

```
workspace
├─ .dockerignore
├─ app.py
├─ config.yaml
├─ core
│  ├─ RAG.py
│  ├─ sql.py
│  ├─ SQL_NS.py
│  └─ sql_oldcode.py
├─ Dockerfile
├─ oracle
│  ├─ instantclient_23_7
│  │  ├─ adrci
│  │  ├─ BASIC_LICENSE
│  │  ├─ BASIC_README
│  │  ├─ fips.so
│  │  ├─ genezi
│  │  ├─ glogin.sql
│  │  ├─ legacy.so
│  │  ├─ libclntsh.so
│  │  ├─ libclntsh.so.10.1
│  │  ├─ libclntsh.so.11.1
│  │  ├─ libclntsh.so.12.1
│  │  ├─ libclntsh.so.18.1
│  │  ├─ libclntsh.so.19.1
│  │  ├─ libclntsh.so.20.1
│  │  ├─ libclntsh.so.21.1
│  │  ├─ libclntsh.so.22.1
│  │  ├─ libclntsh.so.23.1
│  │  ├─ libclntshcore.so
│  │  ├─ libclntshcore.so.12.1
│  │  ├─ libclntshcore.so.18.1
│  │  ├─ libclntshcore.so.19.1
│  │  ├─ libclntshcore.so.20.1
│  │  ├─ libclntshcore.so.21.1
│  │  ├─ libclntshcore.so.22.1
│  │  ├─ libclntshcore.so.23.1
│  │  ├─ libnnz.so
│  │  ├─ libocci.so
│  │  ├─ libocci.so.10.1
│  │  ├─ libocci.so.11.1
│  │  ├─ libocci.so.12.1
│  │  ├─ libocci.so.18.1
│  │  ├─ libocci.so.19.1
│  │  ├─ libocci.so.20.1
│  │  ├─ libocci.so.21.1
│  │  ├─ libocci.so.22.1
│  │  ├─ libocci.so.23.1
│  │  ├─ libociei.so
│  │  ├─ libocijdbc23.so
│  │  ├─ libsqlplus.so
│  │  ├─ libsqlplusic.so
│  │  ├─ libtfojdbc1.so
│  │  ├─ network
│  │  │  └─ admin
│  │  │     └─ README
│  │  ├─ ojdbc11.jar
│  │  ├─ ojdbc17.jar
│  │  ├─ ojdbc8.jar
│  │  ├─ pkcs11.so
│  │  ├─ sqlplus
│  │  ├─ SQLPLUS_LICENSE
│  │  ├─ SQLPLUS_README
│  │  ├─ ucp.jar
│  │  ├─ ucp11.jar
│  │  ├─ ucp17.jar
│  │  ├─ uidrvci
│  │  └─ xstreams.jar
│  └─ META-INF
│     ├─ MANIFEST.MF
│     ├─ ORACLE_C.RSA
│     └─ ORACLE_C.SF
├─ prompt
│  ├─ new_prompt_rag.py
│  └─ prompt_rag.py
├─ ray_deploy
│  ├─ ray_setup.py
│  └─ ray_utils.py
├─ README.md
├─ requirements.txt
├─ save_utils.py
├─ source
│  └─ source_code.txt
├─ static
│  ├─ chat_styles.css
│  ├─ NS_LOGO_ONLY.svg
│  ├─ test_set.js
│  └─ test_styles.css
├─ templates
│  ├─ chatroom.html
│  ├─ index.html
│  ├─ index_socket.html
│  ├─ index_SSE.html
│  ├─ index_test.html
│  └─ index_test_streaming.html
├─ utils
│  ├─ debug_tracking.py
│  ├─ logger_config.py
│  ├─ summarizer.py
│  ├─ tracking.py
│  ├─ utils.py
│  ├─ utils_format.py
│  ├─ utils_load.py
│  └─ utils_vector.py
└─ vectorize.ipynb

```