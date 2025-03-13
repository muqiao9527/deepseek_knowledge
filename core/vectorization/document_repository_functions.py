# 继续 repositories/document_repository.py 中的其余方法

def get_documents(self, limit: int = 100, offset: int = 0,
                  filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    获取文档列表

    Args:
        limit: 返回的最大结果数
        offset: 分页偏移量
        filters: 筛选条件

    Returns:
        文档列表
    """
    try:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 构建查询条件
            query = "SELECT * FROM documents"
            params = []

            if filters:
                conditions = []
                for key, value in filters.items():
                    if key in ["title", "file_name", "file_extension", "content_type", "status"]:
                        conditions.append(f"{key} = ?")
                        params.append(value)

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

            # 添加分页
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            # 执行查询
            cursor.execute(query, params)
            rows = cursor.fetchall()

            # 转换结果
            documents = []
            for row in rows:
                doc = dict(row)

                # 解析元数据
                if "metadata" in doc and doc["metadata"]:
                    try:
                        doc["metadata"] = json.loads(doc["metadata"])
                    except:
                        doc["metadata"] = {}

                documents.append(doc)

            return documents

    except Exception as e:
        logger.error(f"获取文档列表时出错: {str(e)}")
        return []


def update_document(self, document_id: str, updates: Dict[str, Any]) -> bool:
    """
    更新文档信息

    Args:
        document_id: 文档ID
        updates: 更新的字段

    Returns:
        是否更新成功
    """
    try:
        allowed_fields = ["title", "status", "metadata"]
        update_data = {}

        # 过滤允许更新的字段
        for field in allowed_fields:
            if field in updates:
                if field == "metadata" and isinstance(updates[field], dict):
                    update_data[field] = json.dumps(updates[field])
                else:
                    update_data[field] = updates[field]

        if not update_data:
            logger.warning(f"更新文档 {document_id} 时没有有效字段")
            return False

        # 添加更新时间
        update_data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 构建更新语句
            set_clause = ", ".join([f"{field} = ?" for field in update_data.keys()])
            params = list(update_data.values())
            params.append(document_id)

            query = f"UPDATE documents SET {set_clause} WHERE id = ?"
            cursor.execute(query, params)

            # 检查是否有记录被更新
            if cursor.rowcount > 0:
                conn.commit()
                logger.info(f"更新了文档 {document_id}")
                return True
            else:
                logger.warning(f"文档 {document_id} 不存在或无需更新")
                return False

    except Exception as e:
        logger.error(f"更新文档时出错: {str(e)}")
        return False


def delete_document(self, document_id: str) -> bool:
    """
    删除文档

    Args:
        document_id: 文档ID

    Returns:
        是否删除成功
    """
    try:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 首先删除文档内容和分块
            cursor.execute("DELETE FROM document_contents WHERE document_id = ?", (document_id,))
            cursor.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))

            # 删除文档记录
            cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))

            # 检查是否有记录被删除
            if cursor.rowcount > 0:
                conn.commit()
                logger.info(f"删除了文档 {document_id}")
                return True
            else:
                logger.warning(f"文档 {document_id} 不存在")
                return False

    except Exception as e:
        logger.error(f"删除文档时出错: {str(e)}")
        return False


def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
    """
    获取文档的分块

    Args:
        document_id: 文档ID

    Returns:
        文档分块列表
    """
    try:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM document_chunks WHERE document_id = ? ORDER BY chunk_index",
                (document_id,)
            )

            rows = cursor.fetchall()
            chunks = [dict(row) for row in rows]

            return chunks

    except Exception as e:
        logger.error(f"获取文档分块时出错: {str(e)}")
        return []


def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    简单的文档搜索

    Args:
        query: 搜索关键词
        limit: 返回的最大结果数

    Returns:
        匹配的文档列表
    """
    try:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 简单的文本匹配查询
            cursor.execute(
                """
                SELECT d.* FROM documents d
                JOIN document_contents c ON d.id = c.document_id
                WHERE d.title LIKE ? OR c.content LIKE ?
                ORDER BY d.created_at DESC
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit)
            )

            rows = cursor.fetchall()
            documents = []

            for row in rows:
                doc = dict(row)

                # 解析元数据
                if "metadata" in doc and doc["metadata"]:
                    try:
                        doc["metadata"] = json.loads(doc["metadata"])
                    except:
                        doc["metadata"] = {}

                documents.append(doc)

            return documents

    except Exception as e:
        logger.error(f"搜索文档时出错: {str(e)}")
        return []


def get_statistics(self) -> Dict[str, Any]:
    """
    获取文档存储库统计信息

    Returns:
        统计信息字典
    """
    try:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 获取文档数量
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]

            # 获取分块数量
            cursor.execute("SELECT COUNT(*) FROM document_chunks")
            chunk_count = cursor.fetchone()[0]

            # 获取文件类型统计
            cursor.execute(
                "SELECT file_extension, COUNT(*) FROM documents GROUP BY file_extension"
            )
            file_types = {row[0]: row[1] for row in cursor.fetchall()}

            # 获取总存储大小
            cursor.execute("SELECT SUM(file_size) FROM documents")
            total_size = cursor.fetchone()[0] or 0

            return {
                "document_count": doc_count,
                "chunk_count": chunk_count,
                "file_types": file_types,
                "total_size": total_size,
                "database_path": self.db_path
            }

    except Exception as e:
        logger.error(f"获取统计信息时出错: {str(e)}")
        return {
            "document_count": 0,
            "chunk_count": 0,
            "file_types": {},
            "total_size": 0,
            "database_path": self.db_path,
            "error": str(e)
        }